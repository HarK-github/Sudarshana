"""Build Sudarshana_submission.zip outside the repo.

Includes: reports/FINAL_PROJECT_REPORT.pdf, src/, sudarshana/, scripts/, results/,
models/ (JSON only), requirements.txt, configs, tests, data/splits/DATA_CARD.md.

Excludes: .git/, .venv, __pycache__, *.pyc, P5_HANDOFF.md,
weight files (*.safetensors, *.bin, *.pt), any file > 25 MB.

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
    "src",
    "sudarshana",
    "configs",
    "scripts",
    "tests",
    "reports",
    "results",
    "models",
    "data/splits",
    "ui",
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
# models/ directory: only include JSON files (no .yaml, no weights)
MODELS_JSON_ONLY_EXTS = {".json"}
MAX_BYTES = 25 * 1024 * 1024  # 25 MB
# Files that must never appear in the zip
EXCLUDE_FILENAMES = {"P5_HANDOFF.md"}


def _should_exclude(path: Path, relative_to_repo: Path | None = None) -> bool:
    """Return True if path should be excluded from zip."""
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return True
    if path.name in EXCLUDE_FILENAMES:
        return True
    if path.suffix in EXCLUDE_WEIGHT_EXTS:
        return True
    if path.suffix == ".pyc":
        return True
    # models/ directory: JSON files only
    rel = relative_to_repo or (path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path)
    if str(rel).startswith("models") and path.suffix not in MODELS_JSON_ONLY_EXTS:
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
        rel = p.relative_to(REPO_ROOT) if p.is_relative_to(REPO_ROOT) else p
        if p.exists() and not _should_exclude(p, rel):
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
                rel = fp.relative_to(REPO_ROOT)
                if not _should_exclude(fp, rel):
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
