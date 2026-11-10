"""Recall, false-positive rate, and the Day-1 breakdowns."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def binary_counts(labels: Iterable[int], predictions: Iterable[int]) -> dict[str, int]:
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for label, prediction in zip(labels, predictions, strict=True):
        if label not in (0, 1) or prediction not in (0, 1):
            raise ValueError("labels and predictions must be 0 or 1")
        if label == 1 and prediction == 1:
            counts["tp"] += 1
        elif label == 0 and prediction == 1:
            counts["fp"] += 1
        elif label == 0 and prediction == 0:
            counts["tn"] += 1
        else:
            counts["fn"] += 1
    return counts


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def summarize(labels: list[int], predictions: list[int]) -> dict[str, float | int | None]:
    counts = binary_counts(labels, predictions)
    return {
        **counts,
        "support": len(labels),
        "positives": counts["tp"] + counts["fn"],
        "negatives": counts["tn"] + counts["fp"],
        "recall": _rate(counts["tp"], counts["tp"] + counts["fn"]),
        "fpr": _rate(counts["fp"], counts["fp"] + counts["tn"]),
        "precision": _rate(counts["tp"], counts["tp"] + counts["fp"]),
    }


def grouped_recall(
    rows: list[dict],
    predictions: list[int],
    key: str,
) -> dict[str, dict]:
    buckets: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row, prediction in zip(rows, predictions, strict=True):
        if row["label"] != 1:
            continue
        group = row.get(key)
        name = "none" if group is None else str(group)
        buckets[name].append((1, prediction))
    report = {}
    for name in sorted(buckets):
        labels = [item[0] for item in buckets[name]]
        preds = [item[1] for item in buckets[name]]
        summary = summarize(labels, preds)
        report[name] = {
            "support": summary["positives"],
            "recall": summary["recall"],
            "tp": summary["tp"],
            "fn": summary["fn"],
        }
    return report
