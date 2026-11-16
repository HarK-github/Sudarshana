"""Build Sudarshana_submission.zip outside the repo.

Includes: source, configs, scripts, tests, reports (md/pdf/png/json),
results, README, requirements.txt, data/splits/DATA_CARD.md.

Excludes: .git, __pycache__, .venv, checkpoint weights (*.safetensors/*.bin/*.pt),
any file > 25 MB.

Prints: zip size and file count.
"""

from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ZIP = REPO_ROOT.parent / "Sudarshana_submission.zip"

INCLUDE_DIRS = [
    "sudarshana",
    "configs",
    "scripts",
    "tests",
    "reports",
    "results",
    "data/splits",
]
INCLUDE_ROOT_FILES = [
    "README.md",
    "requirements.txt",
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
]
EXCLUDE_DIRS = {".git", "__pycache__", ".venv", "node_modules", ".mypy_cache", ".pytest_cache"}
EXCLUDE_WEIGHT_EXTS = {".safetensors", ".bin", ".pt"}
MAX_BYTES = 25 * 1024 * 1024  # 25 MB


def _should_exclude(path: Path) -> bool:
    """Return True if path should be excluded from zip."""
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return True
    if path.suffix in EXCLUDE_WEIGHT_EXTS:
        return True
    if path.suffix == ".pyc":
        return True
    try:
        if path.stat().st_size > MAX_BYTES:
            print(f"  SKIP (>25MB): {path.relative_to(REPO_ROOT)}")
            return True
    except OSError:
        return True
    return False


def collect_files() -> list[Path]:
    """Collect all files to include."""
    files: list[Path] = []

    # Root-level single files
    for name in INCLUDE_ROOT_FILES:
        p = REPO_ROOT / name
        if p.exists() and not _should_exclude(p):
            files.append(p)

    # Directories
    for dir_name in INCLUDE_DIRS:
        d = REPO_ROOT / dir_name
        if not d.exists():
            continue
        for root, dirs, fnames in os.walk(d):
            # Prune excluded dirs in-place
            dirs[:] = [dd for dd in dirs if dd not in EXCLUDE_DIRS]
            for fname in fnames:
                fp = Path(root) / fname
                if not _should_exclude(fp):
                    files.append(fp)

    return files


def main() -> None:
    files = collect_files()
    file_count = len(files)

    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fp in files:
            arcname = fp.relative_to(REPO_ROOT.parent)
            zf.write(fp, arcname)

    zip_size_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
    print(f"\nSubmission zip: {OUTPUT_ZIP}")
    print(f"Files included: {file_count}")
    print(f"Zip size:       {zip_size_mb:.2f} MB")


if __name__ == "__main__":
    main()
