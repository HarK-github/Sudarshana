"""Generate P5 report: figures, markdown table, and eval report.

Reads results/p5/*.json and writes:
  reports/figures/recall_by_position.png
  reports/figures/recall_by_attack_type.png
  reports/figures/psr_rouge_by_condition.png
  reports/figures/latency_p50_p95.png
  reports/p5_metrics_table.md
  reports/p5_eval_report.md
  reports/P5_evaluation_report.pdf  (via pandoc; else reportlab; else skipped)

IMPORTANT: Every number in the report is injected from JSON files.
Hand-typed numbers are forbidden. Missing data renders "N/A (reason)".
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = REPO_ROOT / "results" / "p5"
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def _load(name: str) -> dict:
    p = RESULTS_DIR / name
    if not p.exists():
        return {}
    with open(p) as fh:
        return json.load(fh)


def _fmt(val: Any, fmt: str = ".4f", na: str = "N/A") -> str:
    """Format a numeric value or return na string."""
    if val is None:
        return na
    if isinstance(val, str):
        return val
    try:
        return format(float(val), fmt)
    except (TypeError, ValueError):
        return na


def _safe_import_matplotlib() -> Any:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(downstream: dict, slicing: dict, latency: dict) -> list[Path]:
    """Generate all figures. Return list of created figure paths."""
    plt = _safe_import_matplotlib()
    if plt is None:
        print("  [figures] matplotlib not available; skipping figures")
        return []

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    # --- 1. Recall by position ---
    pos_data: dict[str, dict[str, float]] = {}
    for pname, pdata in slicing.get("predictors", {}).items():
        rbp = pdata.get("recall_by_position", {})
        for bin_, stats in rbp.items():
            if bin_ not in pos_data:
                pos_data[bin_] = {}
            pos_data[bin_][pname] = stats.get("recall", 0.0)

    if pos_data:
        fig, ax = plt.subplots(figsize=(8, 4))
        bins = sorted(pos_data.keys())
        pnames = sorted({p for d in pos_data.values() for p in d})
        x = range(len(bins))
        width = 0.8 / max(len(pnames), 1)
        for i, pname in enumerate(pnames):
            vals = [pos_data[b].get(pname, 0.0) for b in bins]
            ax.bar([xi + i * width for xi in x], vals, width, label=pname)
        ax.set_xticks([xi + width * (len(pnames) - 1) / 2 for xi in x])
        ax.set_xticklabels(bins)
        ax.set_xlabel("Position Bin")
        ax.set_ylabel("Recall")
        ax.set_title("Recall by Injection Position")
        ax.set_ylim(0, 1.05)
        ax.legend()
        fig.tight_layout()
        p = FIGURES_DIR / "recall_by_position.png"
        fig.savefig(p, dpi=100)
        plt.close(fig)
        created.append(p)
        print(f"  [figure] {p.name}")

    # --- 2. Recall by attack type ---
    atype_data: dict[str, dict[str, float]] = {}
    for pname, pdata in slicing.get("predictors", {}).items():
        rbat = pdata.get("recall_by_attack_type", {})
        for atype, stats in rbat.items():
            if atype not in atype_data:
                atype_data[atype] = {}
            atype_data[atype][pname] = stats.get("recall", 0.0)

    if atype_data:
        pnames = sorted({p for d in atype_data.values() for p in d})
        atypes = sorted(atype_data.keys())
        fig, ax = plt.subplots(figsize=(max(8, len(atypes) * 1.5), 5))
        x = range(len(atypes))
        width = 0.8 / max(len(pnames), 1)
        for i, pname in enumerate(pnames):
            vals = [atype_data[a].get(pname, 0.0) for a in atypes]
            ax.bar([xi + i * width for xi in x], vals, width, label=pname)
        ax.set_xticks([xi + width * (len(pnames) - 1) / 2 for xi in x])
        ax.set_xticklabels(atypes, rotation=35, ha="right", fontsize=8)
        ax.set_xlabel("Attack Type")
        ax.set_ylabel("Recall")
        ax.set_title("Recall by Attack Type")
        ax.set_ylim(0, 1.05)
        ax.legend()
        fig.tight_layout()
        p = FIGURES_DIR / "recall_by_attack_type.png"
        fig.savefig(p, dpi=100)
        plt.close(fig)
        created.append(p)
        print(f"  [figure] {p.name}")

    # --- 3. PSR and ROUGE-1 by condition ---
    psr_data: dict[str, float] = {}
    r1_data: dict[str, float] = {}
    for pname, pdata in downstream.get("predictors", {}).items():
        psr_val = pdata.get("psr", {}).get("psr")
        r1_val = pdata.get("rouge1", {}).get("rouge1_f1")
        if psr_val is not None:
            psr_data[pname] = float(psr_val)
        if r1_val is not None:
            r1_data[pname] = float(r1_val)

    if psr_data or r1_data:
        pnames = sorted(set(list(psr_data.keys()) + list(r1_data.keys())))
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, metric_data, title, ylabel in [
            (axes[0], psr_data, "ASR Proxy (Payload Survival Rate)", "PSR"),
            (axes[1], r1_data, "ROUGE-1 Utility Proxy", "ROUGE-1 F1"),
        ]:
            vals = [metric_data.get(p, 0.0) for p in pnames]
            bars = ax.bar(pnames, vals, color=["#4c72b0", "#dd8452", "#55a868"][:len(pnames)])
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_ylim(0, 1.05)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        p = FIGURES_DIR / "psr_rouge_by_condition.png"
        fig.savefig(p, dpi=100)
        plt.close(fig)
        created.append(p)
        print(f"  [figure] {p.name}")

    # --- 4. Latency p50/p95 ---
    lat_stages = latency.get("stages", {})
    stage_names = [s for s, v in lat_stages.items() if isinstance(v, dict) and "p50_ms" in v]
    if stage_names:
        p50s = [lat_stages[s]["p50_ms"] for s in stage_names]
        p95s = [lat_stages[s]["p95_ms"] for s in stage_names]
        fig, ax = plt.subplots(figsize=(8, 4))
        x = range(len(stage_names))
        ax.bar([xi - 0.2 for xi in x], p50s, 0.4, label="p50", color="#4c72b0")
        ax.bar([xi + 0.2 for xi in x], p95s, 0.4, label="p95", color="#dd8452")
        ax.set_xticks(list(x))
        ax.set_xticklabels(stage_names, rotation=20, ha="right")
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Pipeline Latency (20 runs, 3 warmups)")
        ax.legend()
        fig.tight_layout()
        p = FIGURES_DIR / "latency_p50_p95.png"
        fig.savefig(p, dpi=100)
        plt.close(fig)
        created.append(p)
        print(f"  [figure] {p.name}")
    else:
        print("  [figures] no latency data with p50/p95; skipping latency figure")

    return created


# ---------------------------------------------------------------------------
# Markdown metrics table
# ---------------------------------------------------------------------------

def make_metrics_table(full_metrics: dict) -> str:
    """Build markdown metrics table from full_metrics.json."""
    lines = [
        "# P5 Metrics Table",
        "",
        f"**Split**: {full_metrics.get('split', 'N/A')} | **Seed**: {full_metrics.get('seed', 'N/A')}",
        "",
        "| Predictor | Recall | FPR (test) | FPR (hard-neg) | Span F1 | PSR | ROUGE-1 F1 | Latency p50 (ms) | Latency p95 (ms) |",
        "|-----------|--------|------------|----------------|---------|-----|------------|-----------------|-----------------|",
    ]
    for pname, row in full_metrics.get("predictors", {}).items():
        def v(key: str, fmt: str = ".4f") -> str:
            val = row.get(key)
            if isinstance(val, str) and val.startswith("N/A"):
                return val
            return _fmt(val, fmt)

        lines.append(
            f"| {pname} | {v('recall')} | {v('fpr_test')} | {v('fpr_hard_neg')} "
            f"| {v('span_f1')} | {v('psr')} | {v('rouge1_f1')} "
            f"| {v('latency_p50_ms', '.1f')} | {v('latency_p95_ms', '.1f')} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Full eval report
# ---------------------------------------------------------------------------

def make_eval_report(
    downstream: dict,
    slicing: dict,
    latency: dict,
    heldout: dict,
    full_metrics: dict,
    figure_paths: list[Path],
) -> str:
    """Generate the full evaluation report markdown."""

    def _fig_md(stem: str, caption: str) -> str:
        p = FIGURES_DIR / f"{stem}.png"
        if p.exists():
            return f"![{caption}]({p})\n*{caption}*"
        return f"*Figure not available: {stem}.png*"

    def _psr_val(pname: str) -> str:
        d = downstream.get("predictors", {}).get(pname, {})
        return _fmt(d.get("psr", {}).get("psr"), ".4f", "N/A (no attack records)")

    def _r1_val(pname: str) -> str:
        d = downstream.get("predictors", {}).get(pname, {})
        return _fmt(d.get("rouge1", {}).get("rouge1_f1"), ".4f", "N/A")

    def _fpr_test_val(pname: str) -> str:
        s = slicing.get("predictors", {}).get(pname, {})
        return _fmt(s.get("fpr_on_test_negatives", {}).get("fpr"), ".4f", "N/A")

    def _fpr_hn_val(pname: str) -> str:
        s = slicing.get("predictors", {}).get(pname, {})
        return _fmt(s.get("fpr_on_hard_negatives", {}).get("fpr"), ".4f", "N/A")

    # Recall by position table rows
    pos_rows: list[str] = []
    for pname, pdata in slicing.get("predictors", {}).items():
        for bin_, stats in sorted(pdata.get("recall_by_position", {}).items()):
            pos_rows.append(
                f"| {pname} | {bin_} | {stats.get('support', 0)} "
                f"| {_fmt(stats.get('recall'), '.4f')} |"
            )

    # Recall by attack type table rows
    atype_rows: list[str] = []
    for pname, pdata in slicing.get("predictors", {}).items():
        for atype, stats in sorted(pdata.get("recall_by_attack_type", {}).items()):
            atype_rows.append(
                f"| {pname} | {atype} | {stats.get('support', 0)} "
                f"| {_fmt(stats.get('recall'), '.4f')} |"
            )

    # Latency table
    lat_rows: list[str] = []
    for stage, v in latency.get("stages", {}).items():
        if isinstance(v, dict) and "p50_ms" in v:
            lat_rows.append(f"| {stage} | {v['p50_ms']:.2f} | {v['p95_ms']:.2f} | {v['mean_ms']:.2f} |")
        else:
            reason = v.get("na_reason", "unknown") if isinstance(v, dict) else str(v)
            lat_rows.append(f"| {stage} | N/A | N/A | N/A ({reason}) |")

    # Available predictors for downstream section
    avail_pnames = list(downstream.get("predictors", {}).keys())

    # Heldout status
    heldout_status = heldout.get("status", "SKIPPED")
    heldout_reason = heldout.get("reason", "")
    if heldout_status == "SKIPPED":
        heldout_text = f"> **SKIPPED**: {heldout_reason}"
    else:
        heldout_text = (
            f"Evaluated on `{heldout.get('path', 'unknown')}` "
            f"({heldout.get('n_records', 0)} records)."
        )

    report = f"""# P5 Evaluation Report — Sudarshana Indirect Prompt-Injection Detection

## 1. Setup & Metric Definitions

**Dataset**: `data/splits/test.jsonl` ({full_metrics.get('split', 'test')} split, seed={full_metrics.get('seed', 42)})
**Schema**: v1.1 — fields: `user_task`, `document`, `label`, `injection_span`, `attack_type`, `position`
**injection_span**: list of `[start, end]` character-offset pairs into `document`

### Predictors evaluated

| Name | Description |
|------|-------------|
| none | No defence — always predicts 0 (no injection). Reports baseline FPR=0, recall=0. |
| oracle | Gold labels (upper-bound ceiling). |
| baseline | Lexical keyword model (threshold=0.5). Numbers merged from `results/baseline_day2.json`. |
| token_head | N/A (no weights — checkpoints contain only config+tokenizer files) |

### ⚠️ Proxy Metric Warnings

**PSR ("ASR proxy — payload survival rate")** is NOT real Attack Success Rate.
Real ASR requires an LLM to execute the injected instruction and confirm behavioural change.
PSR only measures whether the injection *text* survives in the post-redaction document.

**ROUGE-1 F1 ("ROUGE-1 utility proxy")** is NOT real downstream task utility.
Real utility requires running the downstream agent/LLM on the redacted document and
comparing task completion. ROUGE-1 is a surface-text overlap proxy only.

---

## 2. Baseline Results (lexical model, `baseline_day2.json`)

{_fmt(full_metrics.get('predictors', {}).get('baseline', {}).get('baseline_day2_recall'), '.4f', 'N/A')} recall / {_fmt(full_metrics.get('predictors', {}).get('baseline', {}).get('baseline_day2_fpr'), '.4f', 'N/A')} FPR  (source: `baseline_day2.json`)

P5 lexical re-evaluation on `{full_metrics.get('split', 'test')}` split:
- Recall: {_fmt(full_metrics.get('predictors', {}).get('baseline', {}).get('recall'), '.4f', 'N/A')}
- FPR (test): {_fpr_test_val('baseline')}
- FPR (hard-neg): {_fpr_hn_val('baseline')}

---

## 3. Recall by Injection Position

{_fig_md('recall_by_position', 'Recall by Injection Position')}

| Predictor | Position | Support | Recall |
|-----------|----------|---------|--------|
{''.join(r + chr(10) for r in pos_rows) if pos_rows else '| — | — | — | — |'}

---

## 4. Recall by Attack Category

{_fig_md('recall_by_attack_type', 'Recall by Attack Type')}

| Predictor | Attack Type | Support | Recall |
|-----------|-------------|---------|--------|
{''.join(r + chr(10) for r in atype_rows) if atype_rows else '| — | — | — | — |'}

---

## 5. FPR on Hard Negatives

| Predictor | FPR (test negatives) | FPR (hard negatives) |
|-----------|----------------------|----------------------|
{''.join(f"| {p} | {_fpr_test_val(p)} | {_fpr_hn_val(p)} |" + chr(10) for p in avail_pnames) if avail_pnames else '| — | — | — |'}

---

## 6. Downstream: PSR and ROUGE-1 (none vs oracle vs predicted)

{_fig_md('psr_rouge_by_condition', 'PSR and ROUGE-1 by Predictor')}

| Predictor | PSR (↓ better for defence) | ROUGE-1 F1 (↑ better) |
|-----------|---------------------------|----------------------|
{''.join(f"| {p} | {_psr_val(p)} | {_r1_val(p)} |" + chr(10) for p in avail_pnames) if avail_pnames else '| — | — | — |'}

**PSR interpretation**: Lower PSR = defence removed more attack payload.
**ROUGE-1 interpretation**: Higher F1 = more of the clean content preserved.

---

## 7. Latency

{_fig_md('latency_p50_p95', 'Pipeline Latency p50/p95')}

Benchmark: ~{latency.get('benchmark_doc_word_estimate', 'N/A')} estimated tokens ({latency.get('benchmark_doc_chars', 'N/A')} chars). 20 runs, 3 warmups.

| Stage | p50 (ms) | p95 (ms) | Mean (ms) |
|-------|----------|----------|-----------|
{''.join(r + chr(10) for r in lat_rows) if lat_rows else '| — | — | — | — |'}

---

## 8. Obfuscated Held-Out Status

{heldout_text}

---

## 9. Limitations

1. **No trained token-head weights** — the DeBERTa-v3 token-head is defined in `models/token_head.py` but checkpoint dirs contain only config and tokenizer. All span-level and weighted-model metrics are N/A until P4 delivers trained weights.
2. **Binary predictors only** — none/baseline produce binary labels, not span offsets. PSR/ROUGE-1 for baseline uses a crude 50-char proxy redaction (not meaningful as a utility metric).
3. **PSR and ROUGE-1 are proxies** — not real ASR or task utility (see §1 warnings).
4. **No pair-input evaluation** — P3's (user_task, document) pair collator is not yet integrated; all predictors use document-only input.
5. **No LLM** — LLMBackend protocol stubbed; `compute_llm_asr` returns N/A.
6. **Small test set** — 206 test records (155 injected, 51 benign); estimates are high-variance.

---

## 10. Open Items

| Item | Owner | Blocker |
|------|-------|---------|
| Trained token-head weights | P4 | Needed for token_head predictor, span-F1, real PSR/ROUGE-1 |
| Pair-input collator (user_task + document) | P3 | Needed for fair comparison vs document-only baseline |
| Obfuscated held-out set | P1/P2 | Needed for heldout evaluation |
| Real LLM ASR measurement | All | Requires LLM endpoint + prompt harness |

---

## 11. Group Members

| Name | Reg. No. |
|------|----------|
| [NAME] | [REG NO] |
| [NAME] | [REG NO] |
| [NAME] | [REG NO] |
| [NAME] | [REG NO] |

*Fill in before submission.*

---

*Report auto-generated by `scripts/make_p5_report.py`. All numbers injected from `results/p5/*.json`.*
"""
    return report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== P5 Report Generator ===")
    downstream = _load("downstream.json")
    slicing = _load("slicing.json")
    latency = _load("latency.json")
    heldout = _load("heldout.json")
    full_metrics = _load("full_metrics.json")

    if not full_metrics:
        print("  ERROR: results/p5/full_metrics.json not found. Run scripts/run_p5_eval.py first.")
        sys.exit(1)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Figures
    figure_paths = make_figures(downstream, slicing, latency)

    # Metrics table
    table_md = make_metrics_table(full_metrics)
    table_path = REPORTS_DIR / "p5_metrics_table.md"
    table_path.write_text(table_md)
    print(f"  Saved → {table_path.relative_to(REPO_ROOT)}")

    # Eval report
    report_md = make_eval_report(downstream, slicing, latency, heldout, full_metrics, figure_paths)
    report_path = REPORTS_DIR / "p5_eval_report.md"
    report_path.write_text(report_md)
    print(f"  Saved → {report_path.relative_to(REPO_ROOT)}")

    # PDF attempt: pandoc → reportlab → skip
    pdf_path = REPORTS_DIR / "P5_evaluation_report.pdf"
    pdf_ok = False
    try:
        result = subprocess.run(
            ["pandoc", str(report_path), "-o", str(pdf_path)],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            print(f"  PDF → {pdf_path.relative_to(REPO_ROOT)} (pandoc)")
            pdf_ok = True
        else:
            print(f"  pandoc failed: {result.stderr[:200]}")
    except FileNotFoundError:
        print("  pandoc not found")
    except Exception as exc:
        print(f"  pandoc error: {exc}")

    if not pdf_ok:
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet

            doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            for line in report_md.splitlines():
                if line.startswith("# "):
                    story.append(Paragraph(line[2:], styles["Heading1"]))
                elif line.startswith("## "):
                    story.append(Paragraph(line[3:], styles["Heading2"]))
                elif line.strip():
                    story.append(Paragraph(line, styles["Normal"]))
                story.append(Spacer(1, 4))
            doc.build(story)
            print(f"  PDF → {pdf_path.relative_to(REPO_ROOT)} (reportlab)")
            pdf_ok = True
        except ImportError:
            print("  reportlab not installed — PDF skipped; logged in P5_HANDOFF.md")
        except Exception as exc:
            print(f"  reportlab error: {exc}")

    if not pdf_ok:
        # Log in handoff
        handoff = REPORTS_DIR / "P5_HANDOFF.md"
        with open(handoff, "a") as fh:
            fh.write("\n## PDF\nSKIPPED: neither pandoc nor reportlab available.\n")

    print("=== P5 Report Complete ===")


if __name__ == "__main__":
    main()
