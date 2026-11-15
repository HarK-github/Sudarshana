"""P3 Ablation Runner — Sudarshana.

Runs the three ablation conditions and writes a combined JSON report.

Conditions:
  A) chunk_only  — [CLS] chunk [SEP], no augmentation  (baseline)
  B) pair        — [CLS] task [SEP] chunk [SEP], no augmentation
  C) pair_aug    — pair input + obfuscation augmentation (full model)

Each condition runs with seeds [42, 43, 44] and reports mean ± std recall.

Usage:
    python -m sudarshana.training.ablation_runner \\
        --backbone microsoft/deberta-v3-xsmall \\
        --epochs 3 \\
        --output_dir ablation_results
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from typing import Any, Dict, List

from .joint_train import run_training


ABLATION_CONDITIONS = [
    {"name": "chunk_only",  "mode": "chunk_only", "augment": False},
    {"name": "pair",        "mode": "pair",        "augment": False},
    {"name": "pair_aug",    "mode": "pair",        "augment": True},
]

SEEDS = [42, 43, 44]


def _flatten_recall(recall_report: Dict[str, Any]) -> Dict[str, float]:
    """Extract per-category recall values into a flat dict."""
    flat: Dict[str, float] = {}
    for group, cats in recall_report.items():
        for cat, stats in cats.items():
            flat[f"{group}/{cat}"] = stats.get("recall", 0.0)
    return flat


def run_ablations(
    backbone: str = "microsoft/deberta-v3-xsmall",
    epochs: int = 3,
    batch_size: int = 8,
    output_dir: str = "ablation_results",
    seeds: List[int] = None,
) -> Dict[str, Any]:
    if seeds is None:
        seeds = SEEDS

    os.makedirs(output_dir, exist_ok=True)
    summary: Dict[str, Any] = {"conditions": {}}

    for cond in ABLATION_CONDITIONS:
        cond_name = cond["name"]
        print(f"\n{'#'*60}")
        print(f"# ABLATION: {cond_name}")
        print(f"{'#'*60}")

        seed_results: List[Dict[str, float]] = []

        for seed in seeds:
            ckpt_dir = os.path.join(output_dir, f"{cond_name}_seed{seed}")
            metadata = run_training(
                mode=cond["mode"],
                augment=cond["augment"],
                backbone=backbone,
                epochs=epochs,
                batch_size=batch_size,
                seed=seed,
                output_dir=ckpt_dir,
            )
            flat = _flatten_recall(metadata.get("recall_by_category", {}))
            flat["val_loss_final"] = metadata["history"][-1]["val_loss"] if metadata["history"] else None
            seed_results.append(flat)

        # Aggregate across seeds
        all_keys = set()
        for r in seed_results:
            all_keys.update(r.keys())

        agg: Dict[str, Any] = {}
        for key in sorted(all_keys):
            vals = [r[key] for r in seed_results if r.get(key) is not None]
            if vals:
                agg[key] = {
                    "mean": round(statistics.mean(vals), 4),
                    "std":  round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 4),
                    "values": [round(v, 4) for v in vals],
                }

        summary["conditions"][cond_name] = {
            "config": cond,
            "seeds":  seeds,
            "metrics": agg,
        }

    # Write combined report
    report_path = os.path.join(output_dir, "ablation_report.json")
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAblation report → {report_path}")
    _print_summary_table(summary)
    return summary


def _print_summary_table(summary: Dict[str, Any]) -> None:
    """Print a human-readable comparison table."""
    conds = list(summary["conditions"].keys())
    # Collect all metric keys
    all_keys: set = set()
    for cond_data in summary["conditions"].values():
        all_keys.update(cond_data["metrics"].keys())

    print("\n" + "="*72)
    print(f"{'Metric':<40} " + "  ".join(f"{c:<12}" for c in conds))
    print("="*72)
    for key in sorted(all_keys):
        row = f"{key:<40} "
        for c in conds:
            val = summary["conditions"][c]["metrics"].get(key)
            if val:
                row += f"{val['mean']:.4f}±{val['std']:.4f}  "
            else:
                row += f"{'N/A':<14}  "
        print(row)
    print("="*72)


def main() -> None:
    p = argparse.ArgumentParser(description="P3 Ablation Runner")
    p.add_argument("--backbone",   default="microsoft/deberta-v3-xsmall")
    p.add_argument("--epochs",     type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--output_dir", default="ablation_results")
    p.add_argument("--seeds",      nargs="+", type=int, default=SEEDS)
    args = p.parse_args()

    run_ablations(
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        seeds=args.seeds,
    )


if __name__ == "__main__":
    main()
