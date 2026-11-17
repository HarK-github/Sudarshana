"""Held-out / obfuscated set evaluation for P5.

If data/splits has no obfuscated file, this module records SKIPPED
(no obfuscated set yet; P1/P2 deliverable) and does NOT generate one.
Build the file with ``python -m sudarshana.data.build_obfuscated``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sudarshana.data.io import read_jsonl
from sudarshana.data.obfuscation import CLEAN_FAMILY, canonical_sha256

from .downstream import (
    apply_redaction,
    null_spans,
    oracle_spans,
    payload_survival_rate,
    rouge1_retention,
)
from .p5_common import REPO_ROOT, SPLITS_DIR, available_predictors, get_predictor, load_jsonl, wilson_ci
from .slicing import compute_slicing


_OBFUSCATED_CANDIDATES = [
    "obfuscated.jsonl",
    "obfuscated_test.jsonl",
    "heldout_obfuscated.jsonl",
    "test_obfuscated.jsonl",
]

_SKIP_REASON = "no obfuscated set yet; P1/P2 deliverable"


def _portable_path(path: Path) -> str:
    """Repo-relative posix path when the file lives in this checkout."""
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _find_obfuscated(heldout_path: str | None) -> Path | None:
    """Resolve obfuscated split path; return None if not found."""
    if heldout_path and heldout_path != "auto":
        path = Path(heldout_path)
        return path if path.exists() else None
    for name in _OBFUSCATED_CANDIDATES:
        path = SPLITS_DIR / name
        if path.exists():
            return path
    return None


def _paired_by_family(
    records: list[dict],
    child_preds: list[int],
    parent_pred_by_hash: dict[str, int],
) -> dict[str, Any]:
    """Child recall minus parent recall, one block per obfuscation family."""
    buckets: dict[str, list[tuple[int, int]]] = {}
    missing = 0
    for record, child_pred in zip(records, child_preds):
        if int(record.get("label", 0)) != 1:
            continue
        family = str(record.get("obfuscation_family") or "unspecified")
        digest = record.get("parent_sha256")
        parent_pred = parent_pred_by_hash.get(digest) if isinstance(digest, str) else None
        if parent_pred is None:
            missing += 1
            continue
        buckets.setdefault(family, []).append((int(parent_pred), int(child_pred)))

    by_family: dict[str, Any] = {}
    for family, pairs in buckets.items():
        count = len(pairs)
        parent_hits = sum(parent for parent, _ in pairs)
        child_hits = sum(child for _, child in pairs)
        lost = sum(1 for parent, child in pairs if parent == 1 and child == 0)
        gained = sum(1 for parent, child in pairs if parent == 0 and child == 1)
        parent_lo, parent_hi = wilson_ci(parent_hits, count)
        child_lo, child_hi = wilson_ci(child_hits, count)
        lost_lo, lost_hi = wilson_ci(lost, count)
        by_family[family] = {
            "n": count,
            "parent_hits": parent_hits,
            "child_hits": child_hits,
            "parent_recall": round(parent_hits / count, 4),
            "parent_ci": [parent_lo, parent_hi],
            "child_recall": round(child_hits / count, 4),
            "child_ci": [child_lo, child_hi],
            "recall_delta": round((child_hits - parent_hits) / count, 4),
            "lost": lost,
            "gained": gained,
            "lost_rate": round(lost / count, 4),
            "lost_ci": [lost_lo, lost_hi],
        }
    return {"missing_parents": missing, "by_family": by_family}


def _parent_predictions(records: list[dict], parents_by_hash: dict[str, dict], pred_fn: Any) -> dict[str, int]:
    needed: list[tuple[str, dict]] = []
    seen: set[str] = set()
    for record in records:
        if int(record.get("label", 0)) != 1:
            continue
        digest = record.get("parent_sha256")
        if not isinstance(digest, str) or digest in seen or digest not in parents_by_hash:
            continue
        seen.add(digest)
        needed.append((digest, parents_by_hash[digest]))
    if not needed:
        return {}
    predictions = pred_fn([row for _, row in needed])
    return {digest: int(pred) for (digest, _), pred in zip(needed, predictions)}


def format_heldout_section(heldout: dict[str, Any]) -> str:
    """Markdown for the obfuscated held-out section of the P5 reports."""
    status = heldout.get("status", "SKIPPED")
    if status != "OK":
        return f"> **SKIPPED**: {heldout.get('reason', _SKIP_REASON)}"

    lines = [
        f"Evaluated on `{heldout.get('path', 'unknown')}` ({heldout.get('n_records', 0)} records).",
        "",
        "Child recall is the obfuscated payload. Parent recall is the same predictor on the clean test parent. "
        "Delta is child recall minus parent recall. Lost is the fraction of pairs the parent caught and the child missed, "
        "with a Wilson 95% interval. Position and domain slices are in this JSON under each predictor.",
        "",
        "| Predictor | Family | N | Child recall | Parent recall | Delta | Lost [Wilson] |",
        "|-----------|--------|---|--------------|---------------|-------|---------------|",
    ]
    wrote = False
    for name, payload in (heldout.get("predictors") or {}).items():
        if not isinstance(payload, dict):
            continue
        if "error" in payload:
            lines.append(f"| {name} | — | — | — | — | — | error: {payload['error']} |")
            wrote = True
            continue
        for family, block in (payload.get("paired") or {}).get("by_family", {}).items():
            ci = block.get("lost_ci") or [None, None]
            lines.append(
                f"| {name} | {family} | {block.get('n')} | {block.get('child_recall')} | "
                f"{block.get('parent_recall')} | {block.get('recall_delta')} | "
                f"{block.get('lost_rate')} [{ci[0]}, {ci[1]}] |"
            )
            wrote = True
    if not wrote:
        lines.append("")
        lines.append("Paired family table unavailable.")
    return "\n".join(lines)


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
            "reason": _SKIP_REASON,
            "searched_locations": [str(SPLITS_DIR / name) for name in _OBFUSCATED_CANDIDATES],
        }

    records = read_jsonl(obf_path)
    parent_path = SPLITS_DIR / "test.jsonl"
    parents = read_jsonl(parent_path) if parent_path.is_file() else []
    parents_by_hash = {canonical_sha256(row): row for row in parents}
    hard_neg_path = SPLITS_DIR / "hard_negatives.jsonl"
    hard_neg_records = load_jsonl(hard_neg_path) if hard_neg_path.exists() else []

    families = []
    seen_families: set[str] = set()
    for record in records:
        family = record.get("obfuscation_family")
        if family in (None, CLEAN_FAMILY) or family in seen_families:
            continue
        seen_families.add(family)
        families.append(family)

    out: dict[str, Any] = {
        "status": "OK",
        "path": _portable_path(obf_path),
        "n_records": len(records),
        "n_injected": sum(1 for record in records if int(record.get("label", 0)) == 1),
        "n_clean": sum(1 for record in records if int(record.get("label", 0)) == 0),
        "families": families,
        "protocol": "docs/eval_protocol_obfuscated.md",
        "rouge1_reference": "document with gold injection spans removed (the parent context)",
        "predictors": {},
    }

    for name in available_predictors():
        try:
            pred_fn = get_predictor(name)
            preds = pred_fn(records)
            slicing = compute_slicing(records, preds, hard_neg_records, pred_fn)
            parent_preds = _parent_predictions(records, parents_by_hash, pred_fn)
            paired = _paired_by_family(records, preds, parent_preds)

            if name == "none":
                spans = null_spans(records)
            elif name == "oracle":
                spans = oracle_spans(records)
            else:
                spans = null_spans(records)  # binary predictors have no span output

            try:
                redacted = apply_redaction(records, spans)
                psr = payload_survival_rate(records, redacted)
                rouge = rouge1_retention(records, redacted)
            except Exception as exc:
                psr = {"psr": None, "na_reason": str(exc)}
                rouge = {"rouge1_f1": None, "na_reason": str(exc)}

            out["predictors"][name] = {
                "slicing": slicing,
                "paired": paired,
                "psr": psr,
                "rouge1": rouge,
            }
        except Exception as exc:
            out["predictors"][name] = {"error": str(exc)}

    return out
