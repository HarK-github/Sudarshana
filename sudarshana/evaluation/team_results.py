"""P5 — team results ingest, recompute, and reconcile.

load_team_artifacts(root) -> dict
recompute_from_predictions(pred_path, ...) -> dict
reconcile(recomputed, team_reported) -> list[dict]
psr_rouge_from_predictions(pred_path) -> dict

All functions return None / {"status":"N/A","reason":"..."} on missing data;
they never raise on missing files or keys.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(p: Path) -> Optional[dict]:
    try:
        with open(p) as fh:
            return json.load(fh)
    except Exception:
        return None


def _load_jsonl(p: Path) -> Optional[list]:
    try:
        rows = []
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    except Exception:
        return None


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Returns (lo, hi)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def _safe_div(a, b):
    return a / b if b else None


# ---------------------------------------------------------------------------
# 1. load_team_artifacts
# ---------------------------------------------------------------------------

STEP_FILES = {
    "step1": "results/step1_baseline/step1_baseline_seed42_metrics.json",
    "step2": "results/step2_normalizer/step2_normalizer_seed42_metrics.json",
    "step3": "results/step3_chunking/step3_chunking_seed42_metrics.json",
    "step4": "results/step4_pair_input/step4_pair_input_seed42_metrics.json",
    "step5": "results/step5_token_head/step5_token_head_seed42_metrics.json",
    "step6": "results/step6_augmented/step6_augmented_seed42_metrics.json",
}


def load_team_artifacts(root: str | Path) -> dict:
    """Load all team-produced result artifacts.

    Returns a dict with keys: steps, ablation_tables, table_3_5, table_3_8.
    Each entry carries a _source and _owner tag.
    Missing files produce {"status": "N/A", "reason": "file not found"}.
    """
    root = Path(root)
    out: dict[str, Any] = {}

    # Step metrics
    steps: dict[str, Any] = {}
    for step_id, rel in STEP_FILES.items():
        p = root / rel
        d = _load_json(p)
        if d is None:
            steps[step_id] = {"status": "N/A", "reason": f"file not found: {rel}"}
        else:
            d["_source"] = rel
            d["_owner"] = "P3"
            d["_provenance"] = "team-reported"
            steps[step_id] = d
    out["steps"] = steps

    # Ablation ladder tables
    abl_path = root / "reports/ablation_ladder_tables.json"
    abl = _load_json(abl_path)
    if abl is None:
        out["ablation_tables"] = {"status": "N/A", "reason": "reports/ablation_ladder_tables.json not found"}
    else:
        abl["_source"] = "reports/ablation_ladder_tables.json"
        abl["_owner"] = "P3"
        abl["_provenance"] = "team-reported"
        out["ablation_tables"] = abl

    # Table 3.5 obfuscation comparison
    t35_path = root / "results/table_3_5_obfuscation_comparison.json"
    t35 = _load_json(t35_path)
    if t35 is None:
        out["table_3_5"] = {"status": "N/A", "reason": "results/table_3_5_obfuscation_comparison.json not found"}
    else:
        out["table_3_5"] = {
            "data": t35,
            "_source": "results/table_3_5_obfuscation_comparison.json",
            "_owner": "P3",
            "_provenance": "team-reported",
        }

    # Table 3.8 latency
    t38_path = root / "results/table_3_8_latency.json"
    t38 = _load_json(t38_path)
    if t38 is None:
        out["table_3_8"] = {"status": "N/A", "reason": "results/table_3_8_latency.json not found"}
    else:
        out["table_3_8"] = {
            "data": t38,
            "_source": "results/table_3_8_latency.json",
            "_owner": "P3",
            "_provenance": "team-reported",
        }

    return out


# ---------------------------------------------------------------------------
# 2. recompute_from_predictions
# ---------------------------------------------------------------------------

def recompute_from_predictions(
    pred_path: str | Path,
    split_path: Optional[str | Path] = None,
) -> dict:
    """Recompute detection metrics from raw_test_predictions.jsonl.

    The predictions file (reports/audit/raw_test_predictions.jsonl) has fields:
      index, is_attacked, domain, pred_spans, is_blocked, gold_coverage,
      residual_payload, legitimate_retained.

    Uses is_attacked as gold label; is_blocked as predicted label.
    Returns overall + by_position + by_attack_type slices.
    Returns N/A(reason) for slices with no metadata.
    """
    pred_path = Path(pred_path)
    preds = _load_jsonl(pred_path)
    if preds is None:
        return {"status": "N/A", "reason": f"predictions file not found: {pred_path}"}

    # Load split for position/attack_type metadata (optional)
    split_meta: dict[int, dict] = {}
    if split_path is not None:
        rows = _load_jsonl(Path(split_path))
        if rows:
            for i, r in enumerate(rows):
                split_meta[i] = r

    # Build per-record results
    records = []
    for i, p in enumerate(preds):
        gold = int(bool(p.get("is_attacked", False)))
        pred = int(bool(p.get("is_blocked", False)))
        meta = split_meta.get(i, {})
        records.append({
            "gold": gold,
            "pred": pred,
            "position": meta.get("position", "unknown"),
            "attack_type": meta.get("attack_type", "unknown"),
            "domain": p.get("domain", "unknown"),
            "index": p.get("index", i),
        })

    def _compute_metrics(subset: list[dict], n_label: str = "") -> dict:
        tp = sum(1 for r in subset if r["gold"] == 1 and r["pred"] == 1)
        fp = sum(1 for r in subset if r["gold"] == 0 and r["pred"] == 1)
        fn = sum(1 for r in subset if r["gold"] == 1 and r["pred"] == 0)
        tn = sum(1 for r in subset if r["gold"] == 0 and r["pred"] == 0)
        pos = tp + fn
        neg = fp + tn
        recall = _safe_div(tp, pos)
        fpr = _safe_div(fp, neg)
        prec = _safe_div(tp, tp + fp)
        f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
        recall_ci = _wilson_ci(tp, pos) if pos else (None, None)
        fpr_ci = _wilson_ci(fp, neg) if neg else (None, None)
        return {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "n": len(subset),
            "n_pos": pos, "n_neg": neg,
            "recall": recall,
            "recall_ci_lo": recall_ci[0],
            "recall_ci_hi": recall_ci[1],
            "fpr": fpr,
            "fpr_ci_lo": fpr_ci[0],
            "fpr_ci_hi": fpr_ci[1],
            "precision": prec,
            "f1": f1,
            "_provenance": "recomputed (P5, from reports/audit/raw_test_predictions.jsonl)",
        }

    overall = _compute_metrics(records)

    # By position
    by_position: dict[str, Any] = {}
    pos_has_meta = any(r["position"] != "unknown" for r in records)
    if not pos_has_meta and not split_meta:
        by_position = {"status": "N/A", "reason": "position metadata not in predictions file; join split needed"}
    else:
        for pos_val in sorted({r["position"] for r in records}):
            sub = [r for r in records if r["position"] == pos_val]
            # Only recall (attack records)
            attack_sub = [r for r in sub if r["gold"] == 1]
            if not attack_sub:
                by_position[pos_val] = {"status": "N/A", "reason": "no positive records for this position"}
                continue
            tp = sum(1 for r in attack_sub if r["pred"] == 1)
            fn = len(attack_sub) - tp
            recall = _safe_div(tp, tp + fn)
            lo, hi = _wilson_ci(tp, tp + fn)
            by_position[pos_val] = {
                "support": len(attack_sub),
                "recall": recall,
                "recall_ci_lo": lo,
                "recall_ci_hi": hi,
                "_provenance": "recomputed (P5, from raw_test_predictions.jsonl)",
            }

    # By attack_type
    by_attack_type: dict[str, Any] = {}
    at_has_meta = any(r["attack_type"] not in ("unknown", "benign") for r in records)
    if not at_has_meta and not split_meta:
        by_attack_type = {"status": "N/A", "reason": "attack_type not in predictions file; no join data"}
    else:
        for at in sorted({r["attack_type"] for r in records if r["gold"] == 1}):
            sub = [r for r in records if r["gold"] == 1 and r["attack_type"] == at]
            tp = sum(1 for r in sub if r["pred"] == 1)
            fn = len(sub) - tp
            recall = _safe_div(tp, tp + fn)
            lo, hi = _wilson_ci(tp, tp + fn)
            by_attack_type[at] = {
                "support": len(sub),
                "recall": recall,
                "recall_ci_lo": lo,
                "recall_ci_hi": hi,
                "_provenance": "recomputed (P5, from raw_test_predictions.jsonl)",
            }

    return {
        "source": str(pred_path),
        "n": len(records),
        "overall": overall,
        "by_position": by_position,
        "by_attack_type": by_attack_type,
        "_provenance": "recomputed (P5, on data/splits/test.jsonl, n=206)",
    }


# ---------------------------------------------------------------------------
# 3. reconcile
# ---------------------------------------------------------------------------

def reconcile(recomputed: dict, team_reported: dict, threshold: float = 0.005) -> list[dict]:
    """Flag |diff| > threshold on recall/fpr between recomputed and team_reported.

    Returns list of flag dicts (may be empty if no flags).
    Never silently picks one number; always reports both.
    """
    flags = []

    r_overall = recomputed.get("overall", {})
    for step_id, step_data in team_reported.get("steps", {}).items():
        if isinstance(step_data, dict) and step_data.get("status") == "N/A":
            continue
        t_overall = step_data.get("overall", {})
        for metric in ("recall", "fpr"):
            rv = r_overall.get(metric)
            tv = t_overall.get(metric)
            if rv is None or tv is None:
                continue
            diff = abs(rv - tv)
            if diff > threshold:
                flags.append({
                    "step": step_id,
                    "metric": metric,
                    "recomputed": rv,
                    "team_reported": tv,
                    "diff": diff,
                    "flag": f"|diff|={diff:.4f} > {threshold}",
                    "note": "P5 recomputed uses is_blocked as predicted label vs step metrics using threshold-calibrated scores",
                })
    return flags


# ---------------------------------------------------------------------------
# 4. PSR + ROUGE from predictions
# ---------------------------------------------------------------------------

def psr_rouge_from_predictions(pred_path: str | Path) -> dict:
    """Compute PSR and legitimate_retained from raw_test_predictions.jsonl.

    Uses pre-computed fields: residual_payload, legitimate_retained.
    These are real values from the model's predicted spans applied to documents.
    """
    pred_path = Path(pred_path)
    preds = _load_jsonl(pred_path)
    if preds is None:
        return {"status": "N/A", "reason": f"predictions file not found: {pred_path}"}

    rp_vals = [p["residual_payload"] for p in preds if "residual_payload" in p]
    lr_vals = [p["legitimate_retained"] for p in preds if "legitimate_retained" in p]

    if not rp_vals:
        return {"status": "N/A", "reason": "residual_payload field not in predictions"}

    psr = sum(rp_vals) / len(rp_vals)
    lr = sum(lr_vals) / len(lr_vals) if lr_vals else None

    attacked = [p for p in preds if p.get("is_attacked", False)]
    psr_attacked = (
        sum(p["residual_payload"] for p in attacked if "residual_payload" in p) / len(attacked)
        if attacked else None
    )

    return {
        "n": len(preds),
        "n_attacked": len(attacked),
        "psr_overall": psr,
        "psr_attacked_only": psr_attacked,
        "legitimate_retained_mean": lr,
        "_provenance": "recomputed (P5, from reports/audit/raw_test_predictions.jsonl, n=206)",
        "_note": (
            "PSR = payload survival rate proxy for ASR. "
            "Real ASR requires LLM execution. "
            "legitimate_retained = fraction of non-payload chars surviving redaction."
        ),
    }
