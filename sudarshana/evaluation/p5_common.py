"""Common loaders, predictor registry, and helpers for P5 evaluation.

ASSUMPTIONS logged in reports/P5_HANDOFF.md:
- injection_span is list of [start, end] int pairs
- position field: 'start'|'middle'|'end'|'none' (taken as-is from records)
- baseline = lexical keyword model (baseline_day2.json); no baseline.py exists
- token_head: SKIPPED(no weights) — checkpoints have no pytorch_model.bin
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Callable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = REPO_ROOT / "data" / "splits"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
RESULTS_DIR = REPO_ROOT / "results"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    """Load all records from a JSONL file."""
    records: list[dict] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_split(name: str) -> list[dict]:
    """Load a named split (test | val | train | hard_negatives)."""
    return load_jsonl(SPLITS_DIR / f"{name}.jsonl")


# ---------------------------------------------------------------------------
# Span helpers
# ---------------------------------------------------------------------------

def spans_from_record(record: dict) -> list[tuple[int, int]]:
    """Return injection spans as (start, end) tuples; empty list for negatives."""
    raw = record.get("injection_span") or []
    out: list[tuple[int, int]] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            out.append((int(item[0]), int(item[1])))
        elif isinstance(item, dict):
            out.append((int(item["start"]), int(item["end"])))
    return out


def gold_text(record: dict) -> str:
    """Return concatenated injection text from gold spans."""
    doc = record["document"]
    parts = [doc[s:e] for s, e in spans_from_record(record)]
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Position bin helper
# ---------------------------------------------------------------------------

def position_bin(record: dict) -> str:
    """Return position bin: start | middle | end | none.

    Uses the existing 'position' field when present (values already match).
    Falls back to computing from char_start vs doc thirds.
    """
    pos = record.get("position")
    if pos in ("start", "middle", "end", "none"):
        return str(pos)
    spans = spans_from_record(record)
    if not spans:
        return "none"
    doc_len = len(record["document"])
    if doc_len == 0:
        return "none"
    char_start = spans[0][0]
    third = doc_len / 3
    if char_start < third:
        return "start"
    if char_start < 2 * third:
        return "middle"
    return "end"


# ---------------------------------------------------------------------------
# Lexical baseline scorer (implements keyword approach matching baseline_day2)
# ---------------------------------------------------------------------------

_INJECTION_KEYWORDS: list[str] = [
    "ignore", "disregard", "forget", "pretend", "override", "bypass",
    "instead", "instruction", "instructions", "prompt", "system",
    "jailbreak", "do not", "don't", "must", "you are now", "new task",
    "stop", "execute", "admin", "root", "sudo",
]
_KEYWORD_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _INJECTION_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


def _lexical_score(doc: str) -> float:
    """Return normalised keyword match count as a score in [0, 1]."""
    hits = len(_KEYWORD_RE.findall(doc))
    return min(hits / 5.0, 1.0)


# ---------------------------------------------------------------------------
# Predictor registry
# ---------------------------------------------------------------------------

_NA_REASONS: dict[str, str] = {}
_PREDICTORS: dict[str, Callable[[list[dict]], list[int]]] = {}


def _register(name: str, fn: Callable[[list[dict]], list[int]]) -> None:
    _PREDICTORS[name] = fn


def _skip(name: str, reason: str) -> None:
    _NA_REASONS[name] = reason


# --- none ---
def _predict_none(records: list[dict]) -> list[int]:
    """No defence: always predict positive (injection present)."""
    return [0] * len(records)


_register("none", _predict_none)


# --- oracle ---
def _predict_oracle(records: list[dict]) -> list[int]:
    """Oracle: return gold labels (upper-bound ceiling)."""
    return [int(r["label"]) for r in records]


_register("oracle", _predict_oracle)


# --- baseline (lexical keyword model) ---
_LEXICAL_THRESHOLD = 0.5


def _predict_baseline(records: list[dict]) -> list[int]:
    """Lexical keyword baseline — no training required."""
    preds: list[int] = []
    for r in records:
        score = _lexical_score(r["document"])
        preds.append(1 if score >= _LEXICAL_THRESHOLD else 0)
    return preds


_register("baseline", _predict_baseline)


# --- token_head ---
def _check_token_head() -> bool:
    """Return True if any checkpoint dir contains weight files."""
    weight_exts = {".bin", ".safetensors", ".pt"}
    for ckpt_dir in CHECKPOINTS_DIR.iterdir():
        if not ckpt_dir.is_dir():
            continue
        for f in ckpt_dir.iterdir():
            if f.suffix in weight_exts:
                return True
    return False


if _check_token_head():
    def _predict_token_head(records: list[dict]) -> list[int]:
        """Token-head DeBERTa predictor using saved checkpoint weights."""
        from sudarshana.redaction.redact import load_prototype_model, predict_spans
        load_prototype_model()
        preds: list[int] = []
        for r in records:
            spans = predict_spans(r["document"], r.get("user_task"))
            preds.append(1 if spans else 0)
        return preds

    _register("token_head", _predict_token_head)
else:
    _skip("token_head", "no weights — checkpoints/ contain only config+tokenizer files")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def available_predictors() -> list[str]:
    """Return names of all registered (runnable) predictors."""
    return list(_PREDICTORS.keys())


def skipped_predictors() -> dict[str, str]:
    """Return {name: reason} for all skipped predictors."""
    return dict(_NA_REASONS)


def get_predictor(name: str) -> Callable[[list[dict]], list[int]]:
    """Return callable predictor by name; raises KeyError if skipped/unknown."""
    if name in _PREDICTORS:
        return _PREDICTORS[name]
    if name in _NA_REASONS:
        raise KeyError(f"Predictor '{name}' skipped: {_NA_REASONS[name]}")
    raise KeyError(f"Unknown predictor '{name}'")
