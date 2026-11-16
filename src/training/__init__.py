"""Training and evaluation modules for Sudarshana."""

from .metrics_hooks import (
    pool_chunks,
    aggregate_document_predictions,
    compute_classification_metrics,
    compute_stratified_metrics,
    calibrate_threshold_at_fpr,
    evaluate_at_fixed_recall,
    compute_span_f1,
)

__all__ = [
    "pool_chunks",
    "aggregate_document_predictions",
    "compute_classification_metrics",
    "compute_stratified_metrics",
    "calibrate_threshold_at_fpr",
    "evaluate_at_fixed_recall",
    "compute_span_f1",
]
