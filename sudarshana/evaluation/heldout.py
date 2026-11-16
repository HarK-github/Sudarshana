"""Held-out / obfuscated set evaluation for P5.

If data/splits has no obfuscated file, this module records SKIPPED
(no obfuscated set yet; P1/P2 deliverable) and does NOT generate one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .p5_common import load_jsonl, available_predictors, get_predictor, SPLITS_DIR
from .slicing import compute_slicing
from .downstream import (
    payload_survival_rate,
    rouge1_retention,
    apply_redaction,
    oracle_spans,
    null_spans,
)


_OBFUSCATED_CANDIDATES = [
    "obfuscated.jsonl",
    "obfuscated_test.jsonl",
    "heldout_obfuscated.jsonl",
    "test_obfuscated.jsonl",
]


def _find_obfuscated(heldout_path: str | None) -> Path | None:
    """Resolve obfuscated split path; return None if not found."""
    if heldout_path and heldout_path != "auto":
        p = Path(heldout_path)
        return p if p.exists() else None
    for name in _OBFUSCATED_CANDIDATES:
        p = SPLITS_DIR / name
        if p.exists():
            return p
    return None


def evaluate_heldout(path: str | None = "auto") -> dict[str, Any]:
    """Evaluate all available predictors on the held-out obfuscated set.

    Args:
        path: Explicit path to JSONL file, 'auto' to search data/splits/,
              or None to skip.

    Returns:
        Dict with evaluation results or SKIPPED status.
    """
    obf_path = _find_obfuscated(path)
    if obf_path is None:
        return {
            "status": "SKIPPED",
            "reason": "no obfuscated set yet; P1/P2 deliverable",
            "searched_locations": [str(SPLITS_DIR / c) for c in _OBFUSCATED_CANDIDATES],
        }

    records = load_jsonl(obf_path)
    hard_neg_path = SPLITS_DIR / "hard_negatives.jsonl"
    hard_neg_records = load_jsonl(hard_neg_path) if hard_neg_path.exists() else []

    out: dict[str, Any] = {
        "status": "OK",
        "path": str(obf_path),
        "n_records": len(records),
        "predictors": {},
    }

    for name in available_predictors():
        try:
            pred_fn = get_predictor(name)
            preds = pred_fn(records)
            slicing = compute_slicing(records, preds, hard_neg_records, pred_fn)

            # downstream
            if name == "none":
                spans = null_spans(records)
            elif name == "oracle":
                spans = oracle_spans(records)
            else:
                spans = null_spans(records)  # binary predictors have no span output

            redacted = apply_redaction(records, spans)
            psr = payload_survival_rate(records, redacted)
            r1 = rouge1_retention(records, redacted)

            out["predictors"][name] = {
                "slicing": slicing,
                "psr": psr,
                "rouge1": r1,
            }
        except Exception as exc:
            out["predictors"][name] = {"error": str(exc)}

    return out
