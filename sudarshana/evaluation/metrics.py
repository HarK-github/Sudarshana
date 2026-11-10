"""Evaluation metrics for token-level injection detection and span redaction."""

from typing import Dict, List, Tuple, Any
import numpy as np


def compute_token_metrics(
    flat_preds: List[int],
    flat_golds: List[int]
) -> Dict[str, float]:
    """
    Compute token-level precision, recall, and F1 ignoring -100 padding/special tokens.
    Evaluates both binary injection detection (B-INJ/I-INJ vs O) and per-label metrics.
    """
    valid_pairs = [(p, g) for p, g in zip(flat_preds, flat_golds) if g != -100]
    if not valid_pairs:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    preds, golds = zip(*valid_pairs)

    # Binary injection metrics: positive = label in {1, 2}
    tp = sum(1 for p, g in zip(preds, golds) if p in (1, 2) and g in (1, 2))
    fp = sum(1 for p, g in zip(preds, golds) if p in (1, 2) and g == 0)
    fn = sum(1 for p, g in zip(preds, golds) if p == 0 and g in (1, 2))
    tn = sum(1 for p, g in zip(preds, golds) if p == 0 and g == 0)

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = (2 * prec * rec) / max(prec + rec, 1e-8)

    # Per-class F1
    per_class = {}
    for label_id, name in [(0, "O"), (1, "B-INJ"), (2, "I-INJ")]:
        c_tp = sum(1 for p, g in zip(preds, golds) if p == label_id and g == label_id)
        c_fp = sum(1 for p, g in zip(preds, golds) if p == label_id and g != label_id)
        c_fn = sum(1 for p, g in zip(preds, golds) if p != label_id and g == label_id)
        c_prec = c_tp / max(c_tp + c_fp, 1)
        c_rec = c_tp / max(c_tp + c_fn, 1)
        c_f1 = (2 * c_prec * c_rec) / max(c_prec + c_rec, 1e-8)
        per_class[f"{name}_f1"] = round(c_f1, 4)

    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        **per_class
    }


def compute_span_f1(
    predicted_spans_list: List[List[Tuple[int, int]]],
    gold_spans_list: List[List[Tuple[int, int]]],
    iou_threshold: float = 0.0,
    exact_match: bool = False
) -> Dict[str, float]:
    """
    Compute Span-level Precision, Recall, and F1.
    Exact match: predicted (s, e) == gold (s, e).
    Overlap match: intersection / union > iou_threshold (or any character overlap if 0.0).
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for preds, golds in zip(predicted_spans_list, gold_spans_list):
        matched_golds = set()

        for p_start, p_end in preds:
            match_found = False
            for g_idx, (g_start, g_end) in enumerate(golds):
                if exact_match:
                    if p_start == g_start and p_end == g_end:
                        match_found = True
                        matched_golds.add(g_idx)
                        break
                else:
                    # Overlap rule: intersection over union or positive intersection
                    inter = max(0, min(p_end, g_end) - max(p_start, g_start))
                    union = max(p_end, g_end) - min(p_start, g_start)
                    iou = inter / max(union, 1)
                    if (iou_threshold == 0.0 and inter > 0) or (iou > iou_threshold):
                        match_found = True
                        matched_golds.add(g_idx)
                        break

            if match_found:
                total_tp += 1
            else:
                total_fp += 1

        total_fn += len(golds) - len(matched_golds)

    prec = total_tp / max(total_tp + total_fp, 1)
    rec = total_tp / max(total_tp + total_fn, 1)
    f1 = (2 * prec * rec) / max(prec + rec, 1e-8)

    prefix = "exact" if exact_match else "overlap"
    return {
        f"span_{prefix}_precision": round(prec, 4),
        f"span_{prefix}_recall": round(rec, 4),
        f"span_{prefix}_f1": round(f1, 4)
    }


def compute_redaction_quality(
    document: str,
    gold_spans: List[Tuple[int, int]],
    redacted_document: str,
    is_blocked: bool
) -> Dict[str, float]:
    """
    Measure redaction quality:
    - residual_payload_rate: Fraction of injected characters still present in redacted text.
    - document_retained: Fraction of original document preserved.
    """
    doc_len = max(len(document), 1)

    if is_blocked:
        return {
            "residual_payload_rate": 0.0,
            "document_retained": 0.0
        }

    # Document retained
    retained_ratio = len(redacted_document) / doc_len

    # Residual payload rate
    total_inj_chars = 0
    residual_chars = 0

    for s, e in gold_spans:
        inj_text = document[s:e]
        total_inj_chars += len(inj_text)
        if inj_text in redacted_document:
            residual_chars += len(inj_text)
        else:
            # Check partial fragment residual
            # Sample 4-gram sliding check
            fragment_len = min(20, len(inj_text))
            if inj_text[:fragment_len] in redacted_document or inj_text[-fragment_len:] in redacted_document:
                residual_chars += len(inj_text) // 2

    res_rate = residual_chars / max(total_inj_chars, 1) if total_inj_chars > 0 else 0.0

    return {
        "residual_payload_rate": round(res_rate, 4),
        "document_retained": round(retained_ratio, 4)
    }

