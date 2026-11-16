"""Evaluation hooks, chunk pooling, threshold calibration, and stratified metric reporting."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


def score_chunk(
    cls_prob: float,
    tok_inj_probs: Optional[Sequence[float]] = None,
    valid_doc_mask: Optional[Sequence[bool]] = None,
    alpha_gate: float = 0.65,
) -> float:
    """
    Gates a chunk's CLS probability with token-level injection probability.
    Suppresses clean false alarms when token head predicts Benign (O).
    """
    if tok_inj_probs is None or len(tok_inj_probs) <= 1:
        return float(cls_prob)

    arr = np.array(tok_inj_probs, dtype=float)
    if valid_doc_mask is not None and len(valid_doc_mask) == len(arr):
        mask = np.array(valid_doc_mask, dtype=bool)
        if mask.any():
            doc_tok_probs = np.sort(arr[mask])[::-1]
        else:
            doc_tok_probs = np.sort(arr[1:])[::-1]
    else:
        # Ignore index 0 ([CLS])
        doc_tok_probs = np.sort(arr[1:])[::-1]

    if len(doc_tok_probs) == 0:
        return float(cls_prob)

    top_k_tok = float(np.mean(doc_tok_probs[: min(3, len(doc_tok_probs))]))
    return float(np.clip(alpha_gate * cls_prob + (1.0 - alpha_gate) * top_k_tok, 0.0, 1.0))


def aggregate_chunks_to_document(
    chunk_scores: Sequence[float],
    pooling_mode: str = "top2_damped",
    length_penalty: float = 0.015,
) -> float:
    """
    Aggregates chunk scores into a document score with optional length damping.
    """
    if not chunk_scores:
        return 0.0
    if len(chunk_scores) == 1 or pooling_mode == "single":
        return float(chunk_scores[0])

    sorted_scores = sorted(chunk_scores, reverse=True)
    if pooling_mode == "max":
        raw_score = sorted_scores[0]
    elif pooling_mode in {"mean", "avg"}:
        raw_score = sum(sorted_scores) / len(sorted_scores)
    else:
        # Default top2 damped: 0.75 * top1 + 0.25 * top2
        top1 = sorted_scores[0]
        top2 = sorted_scores[1] if len(sorted_scores) > 1 else top1
        raw_score = 0.75 * top1 + 0.25 * top2

    if pooling_mode in {"top2_damped", "penalized_top2", "top_k_penalized"}:
        damped_score = raw_score - length_penalty * math.log(max(1, len(chunk_scores)))
    else:
        damped_score = raw_score

    return float(np.clip(damped_score, 0.0, 1.0))


def pool_chunks(scores: Sequence[float], method: str = "max", k: int = 2) -> float:
    """
    P2 pooling hook: Aggregates multiple chunk-level probability scores into one document-level score.
    """
    if not scores:
        return 0.0
    if method == "single":
        return float(scores[0])
    if method in {"top2_damped", "penalized_top2", "top_k_penalized"}:
        return aggregate_chunks_to_document(scores, pooling_mode="top2_damped")
    if method == "max":
        return float(max(scores))
    if method == "mean":
        return float(sum(scores) / len(scores))
    if method == "top_k":
        sorted_scores = sorted(scores, reverse=True)
        top_k_scores = sorted_scores[: min(k, len(sorted_scores))]
        return float(sum(top_k_scores) / len(top_k_scores))
    return aggregate_chunks_to_document(scores, pooling_mode=method)


def aggregate_document_predictions(
    chunk_predictions: Sequence[Dict[str, Any]],
    pooling_method: str = "max",
    top_k: int = 2,
) -> List[Dict[str, Any]]:
    """
    Groups chunk predictions by doc_id and computes pooled doc_prob.
    Preserves ground-truth metadata from the document chunks.
    """
    docs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in chunk_predictions:
        doc_id = p.get("doc_id") or p.get("sample_id", "unknown")
        docs[doc_id].append(p)

    doc_results: List[Dict[str, Any]] = []
    for doc_id, items in docs.items():
        chunk_probs = []
        for it in items:
            raw_c_prob = float(it.get("chunk_prob", 0.0))
            if pooling_method == "token_gated" or it.get("use_token_gating", False):
                t_probs = it.get("token_probs") or it.get("tok_inj_probs")
                gated_p = score_chunk(raw_c_prob, tok_inj_probs=t_probs)
                chunk_probs.append(gated_p)
            else:
                chunk_probs.append(raw_c_prob)

        pool_mode = "top2_damped" if pooling_method in {"token_gated", "top2_damped", "penalized_top2"} else pooling_method
        doc_prob = pool_chunks(chunk_probs, method=pool_mode, k=top_k)

        # Document ground-truth label: 1 if any chunk is injected, else 0
        doc_label = max(int(it.get("label", 0)) for it in items)

        # Representative metadata
        first = items[0]
        # If any chunk has a specific position bucket (other than none), inherit it
        pos_buckets = [it.get("position_bucket", "none") for it in items if it.get("position_bucket") != "none"]
        pos_bucket = pos_buckets[0] if pos_buckets else first.get("position_bucket", "none")

        cat_list = [it.get("attack_category", "clean") for it in items if it.get("attack_category") not in {"clean", "hard_negative"}]
        cat = cat_list[0] if cat_list else first.get("attack_category", "clean")

        total_latency = sum(float(it.get("inference_latency_ms", 0.0)) for it in items)

        doc_results.append({
            "doc_id": doc_id,
            "sample_id": first.get("sample_id", doc_id),
            "label": doc_label,
            "doc_prob": doc_prob,
            "chunk_probs": chunk_probs,
            "num_chunks": len(items),
            "attack_category": cat,
            "position_bucket": pos_bucket,
            "domain": first.get("domain", "text_qa"),
            "inference_latency_ms": total_latency,
            "chunks": items,
        })

    return doc_results


def compute_classification_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Computes Recall, False Positive Rate (FPR), Precision, and F1 score at a given threshold.
    """
    y_t = np.array(y_true, dtype=int)
    y_p = (np.array(y_prob, dtype=float) >= threshold).astype(int)

    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))
    tn = int(np.sum((y_t == 0) & (y_p == 0)))

    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = float((tp + tn) / len(y_t)) if len(y_t) > 0 else 0.0

    return {
        "recall": recall,
        "fpr": fpr,
        "precision": precision,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total": len(y_t),
        "threshold": float(threshold),
    }


def calibrate_threshold_at_fpr(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    target_fpr: float = 0.01,
) -> float:
    """
    Calibrates threshold tau on negative examples such that FPR <= target_fpr.
    """
    y_t = np.array(y_true, dtype=int)
    y_p = np.array(y_prob, dtype=float)

    neg_scores = y_p[y_t == 0]
    if len(neg_scores) == 0:
        return 0.5

    # Percentile threshold: 1.0 - target_fpr quantile of negatives
    tau = float(np.quantile(neg_scores, max(0.0, min(1.0, 1.0 - target_fpr))))
    return max(0.001, min(0.999, tau))


def evaluate_at_fixed_recall(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    target_recall: float = 0.95,
) -> Dict[str, float]:
    """
    Finds threshold tau such that overall Recall >= target_recall, and computes metrics at that tau.
    """
    y_t = np.array(y_true, dtype=int)
    y_p = np.array(y_prob, dtype=float)

    pos_scores = y_p[y_t == 1]
    if len(pos_scores) == 0:
        return compute_classification_metrics(y_true, y_prob, threshold=0.5)

    # Threshold is the (1 - target_recall) quantile of positive scores
    tau = float(np.quantile(pos_scores, max(0.0, 1.0 - target_recall)))
    metrics = compute_classification_metrics(y_true, y_prob, threshold=tau)
    metrics["calibrated_at_target_recall"] = target_recall
    return metrics


def compute_stratified_metrics(
    predictions: Sequence[Dict[str, Any]],
    threshold: float = 0.5,
    score_key: str = "doc_prob",
) -> Dict[str, Any]:
    """
    Computes all required report breakdowns:
    - Overall Recall & FPR
    - Stratified Recall by Position Bucket (start, middle, end) -> Table 3.3
    - Stratified Recall by Attack Category (task_irrelevant, task_relevant, targeted) -> Table 3.4
    - FPR by Negative Category (clean, hard_negative) -> Table 3.2, Section 4.3
    - Stratified by Domain (text_qa, code_qa) -> Table 3.7
    """
    y_true = [int(p["label"]) for p in predictions]
    y_prob = [float(p.get(score_key, p.get("chunk_prob", 0.0))) for p in predictions]

    overall = compute_classification_metrics(y_true, y_prob, threshold=threshold)

    # 1. Stratified Recall by Position Bucket
    by_position: Dict[str, Dict[str, float]] = {}
    for pos in ["start", "middle", "end"]:
        subset = [p for p in predictions if p.get("position_bucket") == pos and p["label"] == 1]
        if subset:
            sub_t = [int(p["label"]) for p in subset]
            sub_p = [float(p.get(score_key, p.get("chunk_prob", 0.0))) for p in subset]
            sub_m = compute_classification_metrics(sub_t, sub_p, threshold=threshold)
            by_position[pos] = {"recall": sub_m["recall"], "support": len(subset)}
        else:
            by_position[pos] = {"recall": 0.0, "support": 0}

    # 2. Stratified Recall by Attack Category
    by_category: Dict[str, Dict[str, float]] = {}
    for cat in ["task_irrelevant", "task_relevant", "targeted"]:
        subset = [p for p in predictions if p.get("attack_category") == cat and p["label"] == 1]
        if subset:
            sub_t = [int(p["label"]) for p in subset]
            sub_p = [float(p.get(score_key, p.get("chunk_prob", 0.0))) for p in subset]
            sub_m = compute_classification_metrics(sub_t, sub_p, threshold=threshold)
            by_category[cat] = {"recall": sub_m["recall"], "support": len(subset)}
        else:
            by_category[cat] = {"recall": 0.0, "support": 0}

    # 3. FPR by Negative Category
    by_neg_category: Dict[str, Dict[str, float]] = {}
    for cat in ["clean", "hard_negative"]:
        subset = [p for p in predictions if p.get("attack_category") == cat and p["label"] == 0]
        if subset:
            sub_t = [int(p["label"]) for p in subset]
            sub_p = [float(p.get(score_key, p.get("chunk_prob", 0.0))) for p in subset]
            sub_m = compute_classification_metrics(sub_t, sub_p, threshold=threshold)
            by_neg_category[cat] = {"fpr": sub_m["fpr"], "support": len(subset)}
        else:
            by_neg_category[cat] = {"fpr": 0.0, "support": 0}

    # 4. Stratified by Domain (Text QA vs Code QA)
    by_domain: Dict[str, Dict[str, float]] = {}
    for dom in ["text_qa", "code_qa"]:
        subset = [p for p in predictions if p.get("domain") == dom]
        if subset:
            sub_t = [int(p["label"]) for p in subset]
            sub_p = [float(p.get(score_key, p.get("chunk_prob", 0.0))) for p in subset]
            sub_m = compute_classification_metrics(sub_t, sub_p, threshold=threshold)
            by_domain[dom] = {
                "recall": sub_m["recall"],
                "fpr": sub_m["fpr"],
                "f1": sub_m["f1"],
                "support": len(subset),
            }
        else:
            by_domain[dom] = {"recall": 0.0, "fpr": 0.0, "f1": 0.0, "support": 0}

    return {
        "overall": overall,
        "recall_by_position": by_position,
        "recall_by_category": by_category,
        "fpr_by_negative_category": by_neg_category,
        "by_domain": by_domain,
    }


def compute_span_f1(
    gold_spans_list: Sequence[List[Tuple[int, int]]],
    pred_spans_list: Sequence[List[Tuple[int, int]]],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Computes Span-level Precision, Recall, and F1 (P4's evaluation hook).
    A predicted span is a true positive if IoU with a gold span >= iou_threshold.
    """
    total_gold = sum(len(g) for g in gold_spans_list)
    total_pred = sum(len(p) for p in pred_spans_list)

    if total_gold == 0 and total_pred == 0:
        return {"span_precision": 1.0, "span_recall": 1.0, "span_f1": 1.0, "support": 0}

    tp = 0
    for golds, preds in zip(gold_spans_list, pred_spans_list):
        matched_golds = set()
        for p_start, p_end in preds:
            for g_idx, (g_start, g_end) in enumerate(golds):
                if g_idx in matched_golds:
                    continue
                # Calculate Intersection over Union
                inter = max(0, min(p_end, g_end) - max(p_start, g_start))
                union = max(p_end, g_end) - min(p_start, g_start)
                iou = inter / union if union > 0 else 0.0
                if iou >= iou_threshold:
                    tp += 1
                    matched_golds.add(g_idx)
                    break

    prec = float(tp / total_pred) if total_pred > 0 else 0.0
    rec = float(tp / total_gold) if total_gold > 0 else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return {
        "span_precision": prec,
        "span_recall": rec,
        "span_f1": f1,
        "true_positives": tp,
        "total_gold_spans": total_gold,
        "total_pred_spans": total_pred,
    }
