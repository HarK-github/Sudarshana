"""P5 evaluation runner.

Usage:
    python scripts/run_p5_eval.py --split test --heldout auto

Writes:
    results/p5/downstream.json
    results/p5/slicing.json
    results/p5/latency.json
    results/p5/heldout.json
    results/p5/full_metrics.json   (predictor x metric table)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def _save(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"  Saved → {path.relative_to(REPO_ROOT)}")


def _load_baseline_numbers() -> dict:
    """Merge pre-existing baseline numbers from results/baseline_day2.json."""
    p = REPO_ROOT / "results" / "baseline_day2.json"
    if not p.exists():
        return {}
    with open(p) as fh:
        raw = json.load(fh)
    runs = raw.get("runs", {})
    lexical = runs.get("lexical", {})
    splits = lexical.get("splits", {})
    # Prefer test split; fall back to val
    for sname in ("test", "val"):
        if sname in splits:
            overall = splits[sname].get("overall", {})
            return {
                "recall": overall.get("recall"),
                "fpr": overall.get("fpr"),
                "precision": overall.get("precision"),
                "source": f"baseline_day2.json/{sname}",
            }
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="P5 evaluation runner")
    parser.add_argument("--split", default="test", help="Data split to evaluate (default: test)")
    parser.add_argument("--heldout", default="auto", help="Path to obfuscated JSONL or 'auto'")
    args = parser.parse_args()

    from sudarshana.evaluation.p5_common import (
        load_split, available_predictors, skipped_predictors, get_predictor,
        SPLITS_DIR,
    )
    from sudarshana.evaluation.slicing import compute_slicing
    from sudarshana.evaluation.downstream import (
        payload_survival_rate, rouge1_retention,
        apply_redaction, oracle_spans, null_spans,
    )
    from sudarshana.evaluation.latency import run_latency_benchmark
    from sudarshana.evaluation.heldout import evaluate_heldout

    out_dir = REPO_ROOT / "results" / "p5"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== P5 Evaluation | split={args.split} ===")
    records = load_split(args.split)
    hard_neg_records = load_split("hard_negatives")
    print(f"  Records: {len(records)} | Hard-neg: {len(hard_neg_records)}")
    print(f"  Available predictors: {available_predictors()}")
    print(f"  Skipped: {skipped_predictors()}")

    # ---- Downstream + Slicing per predictor ----
    downstream_results: dict = {"predictors": {}}
    slicing_results: dict = {"predictors": {}}

    for name in available_predictors():
        print(f"\n  [predictor={name}]")
        pred_fn = get_predictor(name)
        preds = pred_fn(records)

        # Slicing
        sl = compute_slicing(records, preds, hard_neg_records, pred_fn)
        slicing_results["predictors"][name] = sl

        # Downstream: determine spans for redaction
        if name == "none":
            spans = null_spans(records)
        elif name == "oracle":
            spans = oracle_spans(records)
        else:
            # Binary classifiers: redact entire doc when predicted positive
            spans = []
            for rec, pred in zip(records, preds):
                if pred == 1:
                    # Redact first 50 chars as proxy (no span output for binary)
                    spans.append([(0, min(50, len(rec["document"])))])
                else:
                    spans.append([])

        redacted = apply_redaction(records, spans)
        psr = payload_survival_rate(records, redacted)
        r1 = rouge1_retention(records, redacted)
        downstream_results["predictors"][name] = {"psr": psr, "rouge1": r1}

        print(f"    recall={sl['recall_by_position']} | fpr_test={sl['fpr_on_test_negatives']}")
        print(f"    PSR={psr.get('psr')} | ROUGE-1 F1={r1.get('rouge1_f1')}")

    _save(out_dir / "downstream.json", downstream_results)
    _save(out_dir / "slicing.json", slicing_results)

    # ---- Sensitivity (SIMULATED) ----
    print("\n  [sensitivity analysis — SIMULATED]")
    from sudarshana.evaluation.downstream import sensitivity
    sensitivity_result = sensitivity(
        records,
        keep_fractions=(1.0, 0.9, 0.75, 0.5),
        drop_rates=(0.0, 0.1, 0.25, 0.5),
        seed=SEED,
    )
    _save(out_dir / "sensitivity.json", sensitivity_result)
    print(f"    grid cells={len(sensitivity_result.get('grid', []))}")
    print(f"    oracle attack types={list(sensitivity_result.get('oracle_by_attack_type', {}).keys())}")

    # ---- Latency ----
    print("\n  [latency benchmark]")
    latency = run_latency_benchmark()
    _save(out_dir / "latency.json", latency)
    for stage, v in latency.get("stages", {}).items():
        if isinstance(v, dict) and "p50_ms" in v:
            print(f"    {stage}: p50={v['p50_ms']}ms p95={v['p95_ms']}ms")
        else:
            print(f"    {stage}: {v}")

    # ---- Heldout ----
    print(f"\n  [heldout | path={args.heldout}]")
    heldout = evaluate_heldout(args.heldout)
    _save(out_dir / "heldout.json", heldout)
    print(f"    status={heldout.get('status')} | reason={heldout.get('reason', 'N/A')}")

    # ---- Full metrics table ----
    baseline_day2 = _load_baseline_numbers()
    skipped = skipped_predictors()

    all_predictors = list(available_predictors()) + list(skipped.keys())
    full_table: dict = {
        "split": args.split,
        "seed": SEED,
        "predictors": {},
    }

    for name in all_predictors:
        if name in skipped:
            full_table["predictors"][name] = {
                "recall": f"N/A ({skipped[name]})",
                "fpr": f"N/A ({skipped[name]})",
                "fpr_at_recall": f"N/A ({skipped[name]})",
                "span_f1": f"N/A ({skipped[name]})",
                "psr": f"N/A ({skipped[name]})",
                "rouge1_f1": f"N/A ({skipped[name]})",
                "latency_p50_ms": f"N/A ({skipped[name]})",
                "latency_p95_ms": f"N/A ({skipped[name]})",
            }
            continue

        sl = slicing_results["predictors"].get(name, {})
        ds = downstream_results["predictors"].get(name, {})

        # Overall recall (from attack records)
        pos_bins = sl.get("recall_by_position", {})
        total_tp = sum(v.get("tp", 0) for v in pos_bins.values())
        total_fn = sum(v.get("fn", 0) for v in pos_bins.values())
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else None

        # Merge baseline_day2 for baseline row
        if name == "keyword_heuristic" and baseline_day2:
            # Use baseline_day2 as supplementary reference
            bl_recall = baseline_day2.get("recall")
            bl_fpr = baseline_day2.get("fpr")
        else:
            bl_recall = None
            bl_fpr = None

        fpr_test = sl.get("fpr_on_test_negatives", {}).get("fpr")
        psr = ds.get("psr", {}).get("psr")
        r1f1 = ds.get("rouge1", {}).get("rouge1_f1")

        lat_stages = latency.get("stages", {})
        # Representative latency = redact stage
        redact_lat = lat_stages.get("redact", {})
        p50 = redact_lat.get("p50_ms")
        p95 = redact_lat.get("p95_ms")

        full_table["predictors"][name] = {
            "recall": overall_recall,
            "fpr_test": fpr_test,
            "fpr_hard_neg": sl.get("fpr_on_hard_negatives", {}).get("fpr"),
            "fpr_at_recall": "N/A (binary predictor; no continuous scores)",
            "span_f1": sl.get("span_f1", {}).get("span_f1"),
            "span_f1_na_reason": sl.get("span_f1", {}).get("na_reason"),
            "psr": psr,
            "rouge1_f1": r1f1,
            "latency_p50_ms": p50,
            "latency_p95_ms": p95,
            "baseline_day2_recall": bl_recall,
            "baseline_day2_fpr": bl_fpr,
        }

    _save(out_dir / "full_metrics.json", full_table)
    # Merge sensitivity summary into full_metrics
    full_table["sensitivity"] = {
        "label": sensitivity_result.get("label"),
        "grid": sensitivity_result.get("grid"),
        "oracle_by_attack_type": sensitivity_result.get("oracle_by_attack_type"),
        "oracle_by_position": sensitivity_result.get("oracle_by_position"),
    }
    _save(out_dir / "full_metrics.json", full_table)
    print("\n=== P5 Evaluation Complete ===")
    print(f"  Results in: {out_dir.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
