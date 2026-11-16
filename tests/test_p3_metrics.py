"""Unit tests for P3 metrics hooks, pooling, calibration, and stratification (Task 1.4)."""

import pytest
from src.training.metrics_hooks import (
    pool_chunks,
    aggregate_document_predictions,
    compute_classification_metrics,
    calibrate_threshold_at_fpr,
    evaluate_at_fixed_recall,
    compute_stratified_metrics,
    compute_span_f1,
)


def test_chunk_pooling():
    scores = [0.1, 0.9, 0.4, 0.8]
    assert pool_chunks(scores, method="max") == pytest.approx(0.9)
    assert pool_chunks(scores, method="mean") == pytest.approx((0.1 + 0.9 + 0.4 + 0.8) / 4.0)
    # Top-2 pooling: average of [0.9, 0.8] = 0.85
    assert pool_chunks(scores, method="top_k", k=2) == pytest.approx(0.85)


def test_document_aggregation():
    chunks = [
        {"doc_id": "doc1", "sample_id": "doc1_c0", "label": 0, "chunk_prob": 0.2, "position_bucket": "none", "attack_category": "clean", "domain": "text_qa", "inference_latency_ms": 1.5},
        {"doc_id": "doc1", "sample_id": "doc1_c1", "label": 1, "chunk_prob": 0.85, "position_bucket": "middle", "attack_category": "task_relevant", "domain": "text_qa", "inference_latency_ms": 1.5},
        {"doc_id": "doc2", "sample_id": "doc2_c0", "label": 0, "chunk_prob": 0.05, "position_bucket": "none", "attack_category": "hard_negative", "domain": "text_qa", "inference_latency_ms": 1.0},
    ]

    docs = aggregate_document_predictions(chunks, pooling_method="max")
    assert len(docs) == 2

    d1 = next(d for d in docs if d["doc_id"] == "doc1")
    assert d1["label"] == 1
    assert d1["doc_prob"] == 0.85
    assert d1["position_bucket"] == "middle"
    assert d1["attack_category"] == "task_relevant"
    assert d1["num_chunks"] == 2
    assert d1["inference_latency_ms"] == 3.0

    d2 = next(d for d in docs if d["doc_id"] == "doc2")
    assert d2["label"] == 0
    assert d2["doc_prob"] == 0.05
    assert d2["attack_category"] == "hard_negative"


def test_classification_and_stratification():
    preds = [
        {"doc_prob": 0.9, "label": 1, "position_bucket": "start", "attack_category": "targeted", "domain": "text_qa"},
        {"doc_prob": 0.8, "label": 1, "position_bucket": "middle", "attack_category": "task_relevant", "domain": "code_qa"},
        {"doc_prob": 0.2, "label": 1, "position_bucket": "end", "attack_category": "task_irrelevant", "domain": "text_qa"},
        {"doc_prob": 0.1, "label": 0, "position_bucket": "none", "attack_category": "clean", "domain": "text_qa"},
        {"doc_prob": 0.3, "label": 0, "position_bucket": "none", "attack_category": "hard_negative", "domain": "text_qa"},
    ]

    strat = compute_stratified_metrics(preds, threshold=0.5)
    overall = strat["overall"]
    assert overall["tp"] == 2
    assert overall["fn"] == 1
    assert overall["fp"] == 0
    assert overall["tn"] == 2
    assert overall["recall"] == 2.0 / 3.0
    assert overall["fpr"] == 0.0

    # Recall by position
    by_pos = strat["recall_by_position"]
    assert by_pos["start"]["recall"] == 1.0
    assert by_pos["middle"]["recall"] == 1.0
    assert by_pos["end"]["recall"] == 0.0  # prob 0.2 < 0.5

    # Recall by category
    by_cat = strat["recall_by_category"]
    assert by_cat["targeted"]["recall"] == 1.0
    assert by_cat["task_relevant"]["recall"] == 1.0
    assert by_cat["task_irrelevant"]["recall"] == 0.0

    # FPR on hard negatives vs clean
    by_neg = strat["fpr_by_negative_category"]
    assert by_neg["clean"]["fpr"] == 0.0
    assert by_neg["hard_negative"]["fpr"] == 0.0


def test_calibration_and_fixed_recall():
    y_true = [1, 1, 1, 1, 0, 0, 0, 0]
    y_prob = [0.95, 0.90, 0.85, 0.70, 0.30, 0.20, 0.10, 0.05]

    # Target recall 0.75 should calibrate threshold at ~0.85
    fixed_rec = evaluate_at_fixed_recall(y_true, y_prob, target_recall=0.75)
    assert fixed_rec["recall"] >= 0.75

    # Calibrate threshold at FPR 0.0 (all negatives <= threshold)
    tau = calibrate_threshold_at_fpr(y_true, y_prob, target_fpr=0.0)
    assert tau >= 0.30


def test_span_f1():
    gold = [[(10, 30), (50, 70)]]
    # Exact match for first span, partial overlap (> 0.5 IoU) for second
    pred = [[(10, 30), (52, 68)]]
    res = compute_span_f1(gold, pred, iou_threshold=0.5)
    assert res["span_precision"] == 1.0
    assert res["span_recall"] == 1.0
    assert res["span_f1"] == 1.0

    # Disjoint prediction
    disjoint_pred = [[(100, 120)]]
    res_disjoint = compute_span_f1(gold, disjoint_pred, iou_threshold=0.5)
    assert res_disjoint["span_f1"] == 0.0
