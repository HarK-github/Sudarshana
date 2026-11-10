"""Run 3-seed prototype training suite and ablation without hard negatives."""

import os
import shutil
import json
from sudarshana.training.prototype_train import run_training


def main():
    seeds = [42, 43, 44]
    training_runs = {}

    for seed in seeds:
        out_dir = f"checkpoints/prototype-token-head-deberta-v3-xsmall-seed{seed}"
        print(f"\n>>> Running Prototype Training Seed {seed} <<<")
        run_data = run_training(
            seed=seed,
            epochs=3,
            lr=3e-5,
            batch_size=8,
            include_hard_negatives=True,
            output_dir=out_dir
        )
        training_runs[f"seed_{seed}"] = run_data

    # Ablation: Train without hard negatives on Seed 42
    print(f"\n>>> Running Ablation: Without Hard Negatives (Seed 42) <<<")
    ablation_out_dir = "checkpoints/prototype-token-head-deberta-v3-xsmall-no-hn"
    ablation_data = run_training(
        seed=42,
        epochs=3,
        lr=3e-5,
        batch_size=8,
        include_hard_negatives=False,
        output_dir=ablation_out_dir
    )
    training_runs["ablation_no_hard_negatives"] = ablation_data

    # Copy seed 42 to the canonical prototype checkpoint path
    canonical_dir = "checkpoints/prototype-token-head-deberta-v3-xsmall-20261110"
    if os.path.exists(canonical_dir):
        shutil.rmtree(canonical_dir)
    shutil.copytree("checkpoints/prototype-token-head-deberta-v3-xsmall-seed42", canonical_dir)
    print(f"\nCanonical prototype checkpoint ready at {canonical_dir}")

    # Save comprehensive training log
    os.makedirs("reports", exist_ok=True)
    log_path = "reports/training_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(training_runs, f, indent=2)
    print(f"Logged all runs to {log_path}")


if __name__ == "__main__":
    main()

