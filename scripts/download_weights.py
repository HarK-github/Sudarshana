"""Helper script to download frozen model weights from GitHub Releases.

Usage:
    python scripts/download_weights.py [--step STEP] [--all]
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_SLUG = "sujalsom22/Sudarshana"
TAG = "v1.0.0-weights"

WEIGHT_MAPPINGS = {
    "step1": ("final_step1_baseline", "step1_baseline_weights.pt"),
    "step2": ("final_step2_normalizer", "step2_normalizer_weights.pt"),
    "step3": ("final_step3_chunking", "step3_chunking_weights.pt"),
    "step4": ("final_step4_pair_input", "step4_pair_input_weights.pt"),
    "step5": ("final_step5_token_head", "step5_token_head_weights.pt"),
    "step6": ("final_step6_augmented", "step6_augmented_weights.pt"),
}


def download_weight(folder_name: str, asset_name: str, force: bool = False) -> None:
    dest_path = REPO_ROOT / "models" / folder_name / "model_weights.pt"
    if dest_path.exists() and not force:
        print(f"[-] {dest_path} already exists. Use --force to overwrite.")
        return

    url = f"https://github.com/{REPO_SLUG}/releases/download/{TAG}/{asset_name}"
    print(f"[+] Downloading {asset_name} -> {dest_path.relative_to(REPO_ROOT)}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        def reporthook(count, block_size, total_size):
            if total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                sys.stdout.write(f"\r    Progress: {percent}% ({count * block_size / 1e6:.1f}/{total_size / 1e6:.1f} MB)")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
        print(f"\n[+] Successfully saved {dest_path.name}")
    except Exception as e:
        print(f"\n[!] Failed to download {url}: {e}")
        if dest_path.exists():
            dest_path.unlink()


def main():
    parser = argparse.ArgumentParser(description="Download model weights from GitHub Release")
    parser.add_argument("--step", choices=list(WEIGHT_MAPPINGS.keys()) + ["all"], default="all",
                        help="Which step weights to download (default: all)")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing weights")
    args = parser.parse_args()

    targets = list(WEIGHT_MAPPINGS.keys()) if args.step == "all" else [args.step]
    for k in targets:
        folder, asset = WEIGHT_MAPPINGS[k]
        download_weight(folder, asset, force=args.force)


if __name__ == "__main__":
    main()
