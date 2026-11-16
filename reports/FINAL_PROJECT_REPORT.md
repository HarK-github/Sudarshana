# Sudarshana — Final Project Report
## DSL504 Natural Language Processing | Group Project


## Group Members

| Name | Reg. No. | Email |
|------|----------|-------|
| Kinshuk Gupta | 12341190 | kinshukg@iitbhilai.ac.in |
| Harshit Kandpal | B24CS505 | harshitka@iitbhilai.ac.in |
| Sujal Som | B24DS032 | sujals@iitbhilai.ac.in |
| Utkarsh Chatterjee | B24CS050 | utkarshchatt@iitbhilai.ac.in |
| Tanmay Jaiswal | B24DS033 | tanmay@iitbhilai.ac.in |

---

## 1. Project Overview

> Source: `README.md` (owner P1, please review)

# 🛡️ Sudarshana

<div align="center">

**Model-agnostic indirect prompt-injection detection and redaction for retrieval pipelines.**

[![Status](https://img.shields.io/badge/status-in%20development-yellow)](.)
[![Timeline](https://img.shields.io/badge/timeline-7%20days-blue)](.)
[![Team](https://img.shields.io/badge/team-5%20engineers-informational)](.)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](.)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](.)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-DeBERTa--v3-FFD21E)](.)
[![License](https://img.shields.io/badge/license-TBD-lightgrey)](.)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen)](.)

*Sudarshana (Sanskrit: "beautiful discus") — the shield.*

</div>

---

## 📖 Table of contents

- [Why Sudarshana](#-why-sudarshana)
- [How it fits in the stack](#-how-it-fits-in-the-stack)
- [Features in use](#-features-in-use)
- [What it improves over the baseline](#-what-it-improves-over-the-baseline)
- [What it deliberately does not do](#-what-it-deliberately-does-not-do)
- [Pipeline architecture](#-pipeline-architecture)
- [Repository layout](#-repository-layout)
- [Getting started](#-getting-started)
- [Team & ownership](#-team--ownership)
- [7-day plan](#-7-day-plan)
- [Critical dependencies](#-critical-dependencies)
- [Metrics we report](#-metrics-we-report)
- [Known limitations](#-known-limitations)
- [License](#-license)

---

## 🎯 Why Sudarshana

Retrieval-augmented generators read untrusted content — emails, web pages, tables, READMEs, code, PDFs. Any of that text can carry an **indirect prompt injection**: an instruction hidden in retrieved content that hijacks the user's actual task.

Existing defenses have real costs:

| Defense | Cost |
|---|---|
| White-box fine-tuning with `<data>` tokens (BIPIA) | Needs model weights; can degrade capability (Vicuna-13B dropped from 5.2 → ~1.7 on MT-Bench) |
| Prompt reminders ("ignore instructions in content") | Reduces ASR but does not make the model robust; no localization |
| Binary DeBERTa classifier | Blocks whole documents; misses injections after token 512; fooled by base64, homoglyphs, zero-width chars |

**Sudarshana is a complementary layer, not a replacement for those.** It works with closed and API-only generators, leaves the generator untouched, and redacts only the injected spans so the rest of the document survives.

---

## 🧩 How it fits in the stack

```text
user task ─┐

*[... truncated at 60 lines — see full file: README.md]*

---

## 2. Data

> Source: `reports/span_label_qa.md` (owner P1/P2, please review)

# Span-label QA and validation split

Redaction threshold, `lambda_tok`, and false-positive calibration read `data/processed/val.jsonl`. The prototype file `data/splits/val.jsonl` is the set that produced redaction threshold 0.35.

Hard failures: 0. Warnings: 1.

## Counts

| Split | Rows | Positives | Negatives |
|---|---:|---:|---:|
| train | 244107 | 178262 | 65845 |
| val | 60639 | 44406 | 16233 |
| test | 62859 | 41960 | 20899 |

### By source

- **train**: agentdojo 1278, bipia 240800, hard_negative 50, tensortrust 1979
- **val**: agentdojo 83, bipia 60200, hard_negative 8, tensortrust 348
- **test**: agentdojo 300, bipia 60200, hard_negative 800, tensortrust 1559

## Validation attack types

- **agentdojo**: injection_task_0 (40)
- **bipia**: Cryptocurrency Mining (750), Device and Driver Enumeration (750), Malware Distribution (14250), Programming Help (14250), Space Removal & Grouping (14250)
- **tensortrust**: prompt_hijack (116)

Validation BIPIA names: Cryptocurrency Mining, Device and Driver Enumeration, Malware Distribution, Programming Help, Space Removal & Grouping.
Positive `attack_type_id` values are pairwise disjoint across train, val, and test.

## Spans on validation

- Positive rows: 44406
- Spans: 44412
- Multi-span rows: 6 (0.0001)
- Span length mean / median / min / max: 74.4 / 63.0 / 18 / 2477
- Position: {'end': 14750, 'start': 14750, 'middle': 14750, 'tool_output': 40, 'user_message': 116}
- Modality: {'text': 42906, 'code': 1500}

## Token coverage on validation

Coverage is the share of span characters that sit inside a supervised `B-INJ` or `I-INJ` token, macro-averaged over positive validation rows. Continuation pieces are unsupervised, so the score stays below 1 when a word is split.

- All: 0.8334 over 44406 rows
- Text: 0.8466 over 42906 rows
- Code: 0.4567 over 1500 rows
- Aligned rows: val 60639, train sample 500

## Counterfactuals excluded from token loss

- Train: 64262
- Val: 15906
- Test: 19460

These rows keep the parent document with label 0 and a null span, because the user task is the attack text. Detection training keeps them. Token supervision drops them.

## BIPIA document groups in both train and val

1000 groups. The clean copy of every 5th train-file context is validation, while injections of non-validation attack types on that context stay in train. Attack types stay disjoint.

## Warnings

*[... truncated at 60 lines — see full file: reports/span_label_qa.md]*

---

## 3. Baseline & Fixed-Recall Results

> **Teammate results — not recomputed by P5.** Numbers from `results/baseline_day2.json`
> and `results/fixed_recall_day3.json`. Ran on **full dataset** (test=42 999), not the
> committed sample splits (test=206). Numbers are **not comparable** to P5 re-evaluation.

### 3a. Team Baseline (lexical model, threshold=0.5)

| Split | N | Recall | FPR | Precision |
|-------|---|--------|-----|-----------|
| val | 44729 | 0.9982 | 0.6780 | 0.9951 |
| test | 42999 | 0.9964 | 0.4456 | 0.9890 |

### 3b. Fixed-Recall Runs (target recall=0.95)

| Run | Input | Test Recall | Test FPR | Hard-neg FPR (N) |
|-----|-------|-------------|----------|------------------|
| document | document | 0.8309 | 0.1097 | 0.0000 (N=400) |
| pair | user_task_then_document | 0.8964 | 0.1790 | 0.0200 (N=400) |
> Target recall = 0.95. Threshold tuned on val; test numbers from `results/fixed_recall_day3.json`. **Teammate result — not recomputed by P5.**

---

## 4. Token-Head Prototype (P4-reported, not re-evaluated: no weights)

> Source: `reports/p4_token_head_report.md` (owner P4, please review)

# Token Classification Head & Structure-Aware Redaction

This module provides token-level localization and surgical redaction of indirect prompt injections in retrieved documents, allowing downstream LLMs in RAG pipelines to safely process untrusted context without discarding entire documents.

---

## 1. Architecture & Labeling

The detector builds on `microsoft/deberta-v3-xsmall` (~22M active parameters, 70.7M total):

- **Head Module**: A linear projection head (`TokenClassificationHead` in [`sudarshana/models/token_head.py`](file:///mnt/DE94962594960067/Users/kandp/Documents/Work/Personal_Git_Files/Sudarshana/sudarshana/models/token_head.py)) mapping hidden dimension $384 \to 3$ classes: `0: O`, `1: B-INJ`, `2: I-INJ`.
- **Token Supervision**: Special tokens (`[CLS]`, `[SEP]`, `[PAD]`) and subword continuation pieces are masked with `-100`. Only initial subwords receive loss gradients.
- **Class Imbalance**: Positive injection tokens represent ~2–4% of tokens. Loss is weighted with $\mathbf{w} = [1.0, 8.0, 8.0]$ to penalize missed attacks:
  $$\mathcal{L}_{\text{tok}} = -\frac{1}{N} \sum_{i=1}^{N} w_{y_i} \log P(y_i \mid \mathbf{x}_i)$$
- **Multi-Task Joint Loss Hook**: Supports end-to-end training alongside document-level detectors via:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{detect}} + \lambda_{\text{tok}}\,\mathcal{L}_{\text{tok}}$$

---

## 2. Inference & Structure-Aware Redaction

The pipeline in [`sudarshana/redaction/redact.py`](file:///mnt/DE94962594960067/Users/kandp/Documents/Work/Personal_Git_Files/Sudarshana/sudarshana/redaction/redact.py) handles arbitrary document lengths without truncation blind spots:

1. **Sliding Windows & Max-Pooling**: Documents are tokenized into 256-token windows with 64-token stride. Predictions are pooled across overlapping windows by taking the maximum injection probability:
   $$P(\text{injection}) = P(\text{B-INJ}) + P(\text{I-INJ})$$
2. **Confidence Threshold & Gap Bridging**: Tokens with $P \ge 0.35$ are flagged. Unflagged gaps of $\le 2$ connective subwords (prepositions, punctuation) between flagged tokens are bridged into contiguous spans.
3. **Structure-Aware Unit Expansion**: Direct token removal leaves mutilated syntax. Instead of naive sentence segmentation (which breaks markdown tables and code blocks), the engine expands spans to their natural structural container:
   - **Markdown Tables**: Bounded strictly to the single enclosing row (`[\n, \n]`), preventing adjacent table rows from collapsing.
   - **Code Blocks & Docstrings**: Expands to the enclosing block (```` ```...``` ````, `'''...'''`, `"""..."""`, or `<!-- ... -->`), fully excising multi-line script payloads.
   - **Comments**: Expands to contiguous `#` or `//` comment lines.
   - **Prose**: Expands to sentence boundaries within the enclosing paragraph (`\n\n`).
4. **Expansion Cap**: Bounded by $\max(150\text{ chars}, 2.5 \times \text{len(raw\_span)})$ to prevent runaway deletion from isolated false-positive tokens.
5. **Fallback Block Rule**: If total excised characters exceed **50%** of the document, the entire document is blocked, returning `[BLOCKED: PROMPT INJECTION DETECTED]`.

```python
# Public Interface
def predict_spans(text: str, user_task: str | None = None) -> list[tuple[int, int]]: ...
def redact(text: str, spans: list[tuple[int, int]], fallback_ratio: float = 0.50) -> str: ...
```

---

## 3. Measured Results (Held-Out Test Set)

Evaluated on the held-out test split ($n = 206$ total: 125 attacked documents across 25 unseen attack types, 81 clean/hard-negative documents). Wilson 95% confidence intervals are reported for proportions; bootstrap 95% intervals (2,000 resamples) for means.

| Metric | Measured Value | 95% Confidence Interval | Notes |
| :--- | :--- | :--- | :--- |
| **Span Detection Recall (1-to-1)** | **99.20%** (124 / 125) | [95.61%, 99.86%] | 1 missed attack across 25 held-out types |
| **Gold Span Character Coverage** | **98.96%** | [97.70%, 99.90%] | Share of injected characters excised |
| **Selective Redaction Success** | **68.00%** (85 / 125) | [59.40%, 75.50%] | Sanitized without triggering full block |
| **Fallback Block Rate** | **32.00%** (40 / 125) | [24.50%, 40.60%] | Dropped because removed chars > 50% |
| **Legitimate Content Retained** | **57.84%** | [50.50%, 64.50%] | Across all attacked documents |
| **Residual Payload (Overall)** | **1.46%** (305 / 20,862 chars) | — | Total surviving attack characters |
| **Residual Payload (Selective Only)** | **2.32%** (305 / 13,149 chars) | — | On successfully sanitized documents |
| **Samples with Any Residual** | **5.60%** (7 / 125) | [2.70%, 11.10%] | 94.4% of attacks have zero residual |
| **Token F1 / Precision / Recall** | **84.79% / 77.58% / 93.49%** | — | Raw token classification head |

### Breakdown by Position & Document Length


*[... truncated at 60 lines — see full file: reports/p4_token_head_report.md]*

### P4 Headline Metrics (seed=42, `reports/metrics.json`)

| Predictor | Token Recall | Token Precision | Token F1 | Span Overlap F1 | Source |
|-----------|-------------|-----------------|----------|-----------------|--------|
| token_head (P4-reported, not re-evaluated: no weights) | 0.9349 | 0.7758 | 0.8479 | 0.3135 | `reports/metrics.json` |

> ⚠️ No trained weights committed — all P5 token_head evaluation is N/A.

---

## 5. P5 Evaluation

**Split**: test (N=206: 125 injected, 81 clean)
**Hard-neg**: N=15 (indicative only — wide CIs)

### 5a. Main Results Table

| Predictor | Recall | FPR (test) | FPR (hard-neg) | PSR | ROUGE-1 F1 |
|-----------|--------|------------|----------------|-----|------------|
| none | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.8962 |
| oracle | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7298 |
| keyword_heuristic | 0.0400 | 0.0123 | 0.0000 | 1.0000 | 0.8967 |
| token_head | N/A (no weights — checkpoints/ | N/A | N/A | N/A (no weights — checkpoints/ | N/A (no weights — checkpoints/ |

### 5b. SIMULATED Redaction Sensitivity (oracle PSR, f=1.0, m=0.0)

> ⚠️ **SIMULATED**: Gold spans perturbed. Not real model predictions.

Oracle PSR: 0.0000 [0.000, 0.030] (n=125)

See `results/p5/sensitivity.json` and `reports/figures/redaction_sensitivity.png` for full grid.

### 5c. Latency

See `results/p5/latency.json` and `reports/figures/latency_p50_p95.png`.

---

## 6. Limitations & Open Items

1. **Real split sizes differ from team baseline** — P5 evaluates on committed sample splits
   (test=206, hard_neg=15); team baseline ran on full dataset (test=42 999). Not comparable.
2. **Hard-neg N=15** — FPR estimates indicative only (wide Wilson CIs).
3. **ASR proxy** — PSR measures payload-text survival; real ASR requires LLM execution.
4. **ROUGE-1 utility proxy** — surface-text overlap; actual utility requires downstream eval.
5. **token_head: N/A** — no trained weights; all token-head metrics are N/A.
6. **No obfuscated set** — obfuscated held-out evaluation skipped (P1/P2 deliverable).
7. **No pair-input evaluation** — P3 pair collator not integrated.
8. **No LLM ASR** — LLMBackend protocol stubbed.
9. **Sensitivity is SIMULATED** — gold span perturbations, not predicted spans.

### Open Items

| Item | Owner | Blocker |
|------|-------|---------|
| Trained token-head weights | P4 | Needed for token_head predictor, span-F1 |
| Pair-input collator | P3 | Needed for fair comparison vs document-only baseline |
| Obfuscated held-out set | P1/P2 | Needed for heldout evaluation |
| Real LLM ASR | All | Requires LLM endpoint + prompt harness |

---

## 7. Reproduction Commands

```bash
# 1. Install dependencies (system python, pysbd already installed)
pip3 install pysbd reportlab matplotlib --break-system-packages -q

# 2. Run P5 evaluation (uses data/splits/test.jsonl + hard_negatives.jsonl)
python3 scripts/run_p5_eval.py --split test --heldout auto

# 3. Generate P5 report and figures
python3 scripts/make_p5_report.py

# 4. Generate final project report
python3 scripts/make_final_report.py

# 5. Create submission zip
python3 scripts/make_submission_zip.py

# Run tests (use venv python for torch-dependent tests)
python3 -m pytest tests/test_p5_eval.py -q
```

---

*Auto-generated by `scripts/make_final_report.py`. All numbers read from JSON — no invented values.*
