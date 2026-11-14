"""Shared fixture-corpus builder for schema 1.1 tests."""

from __future__ import annotations

from pathlib import Path

from sudarshana.data.build import build_from_config

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "raw"


def build_fixture_corpus(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    build_from_config(
        {
            "seed": 2023,
            "download": False,
            "raw_dir": str(FIXTURES),
            "output_dir": str(output_dir),
            "bipia": {
                "download": False,
                "tasks": ["email", "code"],
                "positions": ["end", "start", "middle"],
            },
            "agentdojo": {"mode": "jsonl", "held_out_suite": "travel"},
            "tensortrust": {"download": False, "val_fraction": 0.15},
            "hard_negatives": {"download": False, "held_out_category": "shell", "val_fraction": 0.15},
        },
        Path("unused.yaml"),
    )
    return output_dir
