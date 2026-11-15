"""Slicing metrics for P5: recall by position/attack_type, FPR, span-F1.

Reuses sudarshana/evaluation/metrics.py functions where available.
Provides threshold sweep for FPR at fixed recall targets when needed.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .p5_common import load_jsonl, position_bin, spans_from_record, SPLITS_DIR


# ---------------------------------------------------------------------------
# Thin wrappers around existing metrics.py
# ---------------------------------------------------------------------------

def _summarize(labels: list[int], preds: list[int]) -> dict[str, Any]:
    """Binary classification summary (tp, fp, tn, fn, recall, fpr, precision)."""
    tp = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 1)
    fp = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 1)
    tn = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 0)
    fn = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 0)
    positives = tp + fn
    negatives = fp + tn
    recall = tp / positives if positives else 0.0
    fpr = fp / negatives if negatives else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "positives": positives, "negatives": negatives,
        "recall": round(recall, 4),
        "fpr": round(fpr, 4),
        "precision": round(precision, 4),
    }


# ---------------------------------------------------------------------------
# Recall by position bin
# ---------------------------------------------------------------------------

def recall_by_position(
    records: list[dict],
    predictions: list[int],
) -> dict[str, dict[str, Any]]:
    """Group injected records by position bin and compute per-bin recall."""
    buckets: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for rec, pred in zip(records, predictions):
        if int(rec["label"]) != 1:
            continue
        bin_ = position_bin(rec)
        buckets[bin_].append((1, pred))
    out: dict[str, dict[str, Any]] = {}
    for bin_, pairs in sorted(buckets.items()):
        labels = [l for l, _ in pairs]
        preds_ = [p for _, p in pairs]
        s = _summarize(labels, preds_)
        out[bin_] = {"support": len(labels), "recall": s["recall"], "tp": s["tp"], "fn": s["fn"]}
    return out


# ---------------------------------------------------------------------------
# Recall by attack_type
# ---------------------------------------------------------------------------

def recall_by_attack_type(
    records: list[dict],
    predictions: list[int],
) -> dict[str, dict[str, Any]]:
    """Per-attack_type recall on injected records."""
    buckets: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for rec, pred in zip(records, predictions):
        if int(rec["label"]) != 1:
            continue
        atype = str(rec.get("attack_type") or "unknown")
        buckets[atype].append((1, pred))
    out: dict[str, dict[str, Any]] = {}
    for atype, pairs in sorted(buckets.items()):
        labels = [l for l, _ in pairs]
        preds_ = [p for _, p in pairs]
        s = _summarize(labels, preds_)
        out[atype] = {"support": len(labels), "recall": s["recall"], "tp": s["tp"], "fn": s["fn"]}
    return out


# ---------------------------------------------------------------------------
# FPR helpers
# ---------------------------------------------------------------------------

def fpr_on_negatives(
    records: list[dict],
    predictions: list[int],
) -> dict[str, Any]:
    """FPR across all label-0 records in the given record list."""
    neg_labels = [0 for r in records if int(r["label"]) == 0]
    neg_preds = [p for r, p in zip(records, predictions) if int(r["label"]) == 0]
    if not neg_labels:
        return {"fpr": None, "support": 0, "fp": 0, "tn": 0}
    s = _summarize(neg_labels, neg_preds)
    return {"fpr": s["fpr"], "support": s["negatives"], "fp": s["fp"], "tn": s["tn"]}


def fpr_on_hard_negatives(
    hard_neg_records: list[dict],
    predictor_fn: Any,
) -> dict[str, Any]:
    """FPR specifically on the hard_negatives split."""
    preds = predictor_fn(hard_neg_records)
    return fpr_on_negatives(hard_neg_records, preds)


def fpr_on_test_negatives(
    test_records: list[dict],
    predictions: list[int],
) -> dict[str, Any]:
    """FPR on label-0 records within the test split."""
    return fpr_on_negatives(test_records, predictions)


# ---------------------------------------------------------------------------
# FPR at fixed recall (threshold sweep)
# ---------------------------------------------------------------------------

def fpr_at_recall_targets(
    records: list[dict],
    scores: list[float],
    targets: tuple[float, ...] = (0.90, 0.95, 0.99),
) -> dict[str, Any]:
    """Sweep thresholds to find FPR at recall >= target.

    Args:
        records: Evaluation records with 'label' field.
        scores:  Continuous detection score per record (higher = more injected).
        targets: Recall target values to evaluate at.

    Returns:
        Dict mapping str(target) -> {'fpr': float, 'threshold': float, 'actual_recall': float}.
    """
    labels = [int(r["label"]) for r in records]
    # Build sorted threshold candidates
    thresholds = sorted(set(scores), reverse=True)
    result: dict[str, Any] = {}
    for target in targets:
        best: dict[str, Any] | None = None
        for thr in thresholds:
            preds = [1 if s >= thr else 0 for s in scores]
            s = _summarize(labels, preds)
            if s["recall"] >= target:
                best = {
                    "threshold": thr,
                    "actual_recall": s["recall"],
                    "fpr": s["fpr"],
                }
                break
        result[str(target)] = best if best else {"fpr": None, "threshold": None, "actual_recall": None}
    return result


# ---------------------------------------------------------------------------
# Span F1 (character-level, when span predictions are available)
# ---------------------------------------------------------------------------

def _span_overlap_f1(
    pred_spans: list[tuple[int, int]],
    gold_spans: list[tuple[int, int]],
    doc_len: int,
) -> float:
    """Character-level F1 between predicted and gold spans."""
    def to_set(spans: list[tuple[int, int]]) -> set[int]:
        chars: set[int] = set()
        for s, e in spans:
            chars.update(range(s, e))
        return chars

    pred_chars = to_set(pred_spans)
    gold_chars = to_set(gold_spans)
    if not gold_chars and not pred_chars:
        return 1.0
    if not gold_chars or not pred_chars:
        return 0.0
    overlap = len(pred_chars & gold_chars)
    prec = overlap / len(pred_chars)
    rec = overlap / len(gold_chars)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def span_f1(
    records: list[dict],
    pred_spans_list: list[list[tuple[int, int]]],
) -> dict[str, Any]:
    """Mean character-level span F1 across injected records.

    Only call when span predictions are available (token_head enabled).
    Returns None metrics with reason when no injected records found.
    """
    f1s: list[float] = []
    for rec, pred_spans in zip(records, pred_spans_list):
        if int(rec["label"]) != 1:
            continue
        gold_spans = spans_from_record(rec)
        doc_len = len(rec["document"])
        f1s.append(_span_overlap_f1(pred_spans, gold_spans, doc_len))
    if not f1s:
        return {"span_f1": None, "na_reason": "no injected records", "n": 0}
    return {"span_f1": round(sum(f1s) / len(f1s), 4), "n": len(f1s)}


# ---------------------------------------------------------------------------
# Full slicing report
# ---------------------------------------------------------------------------

def compute_slicing(
    records: list[dict],
    predictions: list[int],
    hard_neg_records: list[dict],
    predictor_fn: Any,
) -> dict[str, Any]:
    """Run all slicing metrics and return combined dict."""
    return {
        "recall_by_position": recall_by_position(records, predictions),
        "recall_by_attack_type": recall_by_attack_type(records, predictions),
        "fpr_on_test_negatives": fpr_on_test_negatives(records, predictions),
        "fpr_on_hard_negatives": fpr_on_hard_negatives(hard_neg_records, predictor_fn),
        "span_f1": {"span_f1": None, "na_reason": "span predictions not available for binary predictors", "n": 0},
    }
