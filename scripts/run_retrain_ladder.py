"""Automated end-to-end retraining pipeline for remaining ablation steps and final model freeze."""
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.training.run_ablation import run_ablation, load_yaml_config
import scripts.run_day6_freeze as day6_freeze

def main():
    steps = [
        "configs/step3_chunking.yaml",
        "configs/step4_pair_input.yaml",
        "configs/step5_token_head.yaml",
        "configs/step6_augmented.yaml",
    ]

    for cfg_path in steps:
        print(f"\n=======================================================", flush=True)
        print(f"  RUNNING RETRAIN: {cfg_path}", flush=True)
        print(f"=======================================================\n", flush=True)
        cfg = load_yaml_config(cfg_path)
        run_ablation(config=cfg, seed=42, epochs=4)

    print("\n=======================================================", flush=True)
    print("  RUNNING DAY 6 FREEZE & TABLE COMPILATION", flush=True)
    print("=======================================================\n", flush=True)
    day6_freeze.main()

if __name__ == "__main__":
    main()
