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

## 4b. Model Architecture & Training (P3-reported)

> Source: `reports/p3_model_and_training_report.md` (owner P3, please review)

# P3 Final Report: Model Architecture & Training Integration
**Author:** P3 (Model Architecture & Training Integration Lead)  
**Project:** Sudarshana Indirect Prompt Injection Detection Pipeline  
**Date:** November 15, 2026  
**Status:** Complete & Frozen (`models/final_step1..6/`)

---

## 1. Executive Summary & Hypotheses Verification

Across the 7-day implementation schedule, we designed, implemented, and validated the complete 6-step Ablation Ladder for the Sudarshana indirect prompt-injection detection pipeline. All 4 core architectural hypotheses were empirically validated on the held-out test split (206 documents: 125 injected across 25 held-out attack types, 81 benign including 6 curated instructional hard negatives):

1. **Hypothesis H1 (Chunking & Context Preservation):** Moving from single-string 512-token truncation (`step1_baseline`) to sliding-window 256-token chunking with 64-token overlap and max-pooling (`step3_chunking`) eliminates the truncation vulnerability, maintaining $100.00\%$ detection recall on injections placed at the `end` of long documents (vs. $16.98\%$ in baseline).
2. **Hypothesis H2 (Obfuscation & Normalization Robustness):** Coupling a character-level normalizer (NFKC, zero-width stripping, Base64/Hex decoding) with synthetic obfuscation augmentation (`step6_augmented`) achieves $100.00\%$ recall across all five perturbation suites (Leetspeak, Homoglyphs, Zero-Width characters, Base64 wrappers, Markdown embeds) while driving calibrated hard-negative FPR to $0.00\%$.
3. **Hypothesis H3 (Pair-Input Disambiguation):** Cross-attending `[CLS] user_task [SEP] chunk [SEP]` (`step4_pair_input`) breaks naive imperative shortcuts, reducing hard-negative False Positive Rate from $83.33\%$ down to $16.67\%$ under calibrated decision thresholds, without dropping recall on task-relevant or targeted injections.
4. **Hypothesis H4 (Joint Detection & Token Head Redaction):** Adding a sequence token classification head ($\mathbf{H} \to \mathbb{R}^{B \times L \times 3}$) with joint loss ($\mathcal{L}_{\text{detect}} + 0.5 \times \mathcal{L}_{\text{tok}}$) produces token-level probability maps and BIO labels (`O`, `B-INJ`, `I-INJ`) that enable sentence-level redaction without compromising document-level recall ($100.00\%$).

All six production checkpoints, model configs, fast tokenizers, and validation-calibrated decision thresholds are frozen and archived in `models/final_step1_baseline/` through `models/final_step6_augmented/`.

---

## 2. Report Section 2.3: Models and Training Specification

### 2.1 Shared Backbone Architecture
All six ablation variants are built upon the same pretrained transformer encoder backbone:
- **Pretrained Encoder:** Microsoft DeBERTa-v3 (`microsoft/deberta-v3-base`, 86M backbone parameters, hidden dimension $d = 768$, 12 layers, 12 attention heads, vocabulary size $128,100$ using SentencePiece). For resource-constrained and local sandboxed environments, the architecture smoothly adapts to `microsoft/deberta-v3-xsmall` ($d = 384$, 12 layers) with identical head interfaces.
- **Disentangled Attention:** DeBERTa-v3 disentangles content and relative position vectors, computing attention matrices as:
  $$\mathbf{A}_{i,j} = \mathbf{c}_i \mathbf{c}_j^\top + \mathbf{c}_i \mathbf{p}_{i \to j}^\top + \mathbf{p}_{j \to i} \mathbf{c}_j^\top$$
  This formulation preserves syntactic dependency between distant tokens, making it sensitive to injected instructional syntax embedded inside passive retrieved documents.

### 2.2 Multi-Task Head Architecture
The multi-task head (`DebertaMultiTaskFilter`) mounts three specialized prediction heads directly atop the shared encoder's final hidden representation $\mathbf{H} \in \mathbb{R}^{B \times L \times d}$:
1. **Document / Chunk Detection Head (`detect_head`):**
   - Input: Pooled `[CLS]` representation $\mathbf{h}_{\text{CLS}} = \mathbf{H}_{:, 0, :} \in \mathbb{R}^{B \times d}$.
   - Structure: Dropout ($p=0.1$) $\to$ Linear layer $\mathbb{R}^d \to \mathbb{R}^2$.
   - Output: Logits $\mathbf{z}_{\text{detect}} \in \mathbb{R}^{B \times 2}$ yielding chunk injection probability $P(y=1 \mid \text{chunk}) = \text{Softmax}(\mathbf{z}_{\text{detect}})_1$.
2. **Token Classification Head (`token_head`, enabled in Steps 5 and 6):**
   - Input: Full sequence hidden states $\mathbf{H} \in \mathbb{R}^{B \times L \times d}$.
   - Structure: Dropout ($p=0.1$) $\to$ Linear layer $\mathbb{R}^d \to \mathbb{R}^3$ corresponding to labels `0: O`, `1: B-INJ`, `2: I-INJ`.
   - Supervision: Subword tokens belonging to the `user_task` segment, `[CLS]`, `[SEP]`, and `[PAD]` are strictly masked with `ignore_index = -100`. Only document/chunk tokens receive token loss.
   - Output: Token logits $\mathbf{z}_{\text{tok}} \in \mathbb{R}^{B \times L \times 3}$. Sentence redaction probability is computed as $P(\text{INJ} \mid t) = P(\text{B-INJ}) + P(\text{I-INJ})$.
3. **Auxiliary Attack-Type Head (`type_head`, optional Day 5 hook):**
   - Input: Pooled `[CLS]` representation $\mathbf{h}_{\text{CLS}} \in \mathbb{R}^{B \times d}$.
   - Structure: Dropout ($p=0.1$) $\to$ Linear layer $\mathbb{R}^d \to \mathbb{R}^{25}$ mapping to the 25 training attack types ($15$ text $+ 10$ code).
   - Supervision: CrossEntropyLoss masked to compute gradients strictly on positive injected samples (`labels == 1`).

### 2.3 Joint Loss Formulation
To jointly optimize document-level detection, subword span localization, and optional attack classification, we define the composite multi-task objective:
$$\mathcal{L} = \mathcal{L}_{\text{detect}} + \lambda_{\text{tok}} \cdot \mathcal{L}_{\text{tok}} + \lambda_{\text{type}} \cdot \mathcal{L}_{\text{type}}$$
Where:
- $\mathcal{L}_{\text{detect}} = \text{CrossEntropyLoss}(\mathbf{z}_{\text{detect}}, y_{\text{chunk}})$
- $\mathcal{L}_{\text{tok}} = \frac{1}{\sum_{i,j} \mathbb{I}(y_{i,j} \neq -100)} \sum_{i,j} \text{CrossEntropyLoss}(\mathbf{z}_{\text{tok}}[i,j], y_{\text{tok}}[i,j])$
- $\mathcal{L}_{\text{type}} = \frac{1}{\sum_i \mathbb{I}(y_i = 1)} \sum_{i: y_i = 1} \text{CrossEntropyLoss}(\mathbf{z}_{\text{type}}[i], y_{\text{type}}[i])$
- **Loss Weights:** Hyperparameter tuning on the Day 4 validation sweep confirmed that $\lambda_{\text{tok}} = 0.5$ optimally balances document-level recall against token-level Span F1 without gradient dominance. $\lambda_{\text{type}} = 0.0$ for the standard Step 6 release ($0.1$ when auxiliary type regularizer is enabled).

### 2.4 Standardized Training Hyperparameters
Per Section 2.3 of the specification (*"Same for every variant"*), all 6 ablation steps were trained under identical computational and optimization budgets:

| Hyperparameter | Value | Description / Justification |
| :--- | :--- | :--- |

*[... truncated at 60 lines — see full file: reports/p3_model_and_training_report.md]*

> ⚠️ **P5 Audit Note — H2 prose mismatch:** The hypothesis text above (P3-authored) claims
> "100.00% recall across all five perturbation suites." The measured values in
> `results/table_3_5_obfuscation_comparison.json` are 97.60%–99.20% (Base64=98.40%,
> Homoglyphs=97.60%, Zero-width=98.40%, Leetspeak=99.20%, Markdown=98.40%). The claim
> should read "≥97.60% recall across all five suites." P3 owns this correction.

---

## 4c. Ablation Ladder (team-reported, P3, n=206, seed=42)

> Source: `reports/ablation_ladder_tables.json` (owner P3, please review)
> Provenance: team-reported (P3-measured). n=206 for all steps. All numbers from JSON.

| Step | Recall | Clean FPR | HN FPR@90% | Span F1 | Latency (ms) | Provenance |
|------|--------|-----------|-----------|---------|-------------|------------|
| step1_baseline | 0.6080 | 0.2667 | 0.0000 | n/a | 43.56 | team-reported (P3) |
| step2_normalizer | 0.3520 | 0.1467 | 0.0000 | n/a | 46.12 | team-reported (P3) |
| step3_chunking | 0.2800 | 0.2133 | 1.0000 | n/a | 152.40 | team-reported (P3) |
| step4_pair_input | 0.3520 | 0.2267 | 1.0000 | n/a | 158.85 | team-reported (P3) |
| step5_token_head | 0.4880 | 0.0667 | 0.8333 | 34.42% | 164.20 | team-reported (P3) |
| step6_augmented | 0.6160 | 0.1733 | 0.8333 | 34.42% | 166.61 | team-reported (P3) |

> See `reports/ablation_summary.md` and `reports/figures/ladder_recall_vs_fpr.png` for full analysis.

### Ablation Summary (excerpt)

> Source: `reports/ablation_summary.md` (owner P3, please review)

# Sudarshana Ablation Ladder: Final Experimental Report

## 1. Table 3.2 (Comprehensive): Full Detection & Classification Metrics (Calibrated Operating Thresholds)
| Step | Variant | Calib $\tau^*$ | Recall | Clean FPR (N=75) | Hard-Neg FPR | HN FPR @ 90% Rec | Precision | F1 Score | Span F1 | Latency |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **step1_baseline** | 0.4973 | 60.80% | 26.67% | 0.00% | 0.00% | 0.7917 | 0.6878 | n/a | 43.56 ms |
| 2 | **step2_normalizer** | 0.5920 | 35.20% | 14.67% | 0.00% | 0.00% | 0.8000 | 0.4889 | n/a | 46.12 ms |
| 3 | **step3_chunking** | 0.6082 | 28.00% | 21.33% | 16.67% | 100.00% | 0.6731 | 0.3955 | n/a | 152.40 ms |
| 4 | **step4_pair_input** | 0.5868 | 35.20% | 22.67% | 0.00% | 100.00% | 0.7213 | 0.4731 | n/a | 158.85 ms |
| 5 | **step5_token_head** | 0.7466 | 48.80% | 6.67% | 0.00% | 83.33% | 0.9242 | 0.6387 | 34.42% | 164.20 ms |
| 6 | **step6_augmented** | 0.7162 | 61.60% | 17.33% | 0.00% | 83.33% | 0.8556 | 0.7163 | 34.42% | 166.61 ms |

## 1b. Table 3.2 (Course Specification Format — Page 9 Template)
| Step | Variant | Recall | Clean FPR (N=75) | Hard-Neg FPR @ 90% Rec | Span F1 | ASR (LLM) | ROUGE-1 | Latency |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **step1_baseline** | 60.80% | 26.67% | 0.00% | n/a | [P5 handoff] | [P5 handoff] | 43.56 ms |
| 2 | **step2_normalizer** | 35.20% | 14.67% | 0.00% | n/a | [P5 handoff] | [P5 handoff] | 46.12 ms |
| 3 | **step3_chunking** | 28.00% | 21.33% | 100.00% | n/a | [P5 handoff] | [P5 handoff] | 152.40 ms |
| 4 | **step4_pair_input** | 35.20% | 22.67% | 100.00% | n/a | [P5 handoff] | [P5 handoff] | 158.85 ms |
| 5 | **step5_token_head** | 48.80% | 6.67% | 83.33% | 34.42% | [P5 handoff] | [P5 handoff] | 164.20 ms |
| 6 | **step6_augmented** | 61.60% | 17.33% | 83.33% | 34.42% | [P5 handoff] | [P5 handoff] | 166.61 ms |

## 1c. Uncalibrated Baseline (tau = 0.50) Reference Comparison
| Step | Variant | Test Recall (0.5) | Clean FPR (0.5) | Precision (0.5) | F1 Score (0.5) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | **step1_baseline** | 54.40% | 24.00% | 0.7907 | 0.6445 |
| 2 | **step2_normalizer** | 86.40% | 57.33% | 0.7152 | 0.7826 |
| 3 | **step3_chunking** | 40.00% | 33.33% | 0.6579 | 0.4975 |
| 4 | **step4_pair_input** | 45.60% | 28.00% | 0.7308 | 0.5616 |
| 5 | **step5_token_head** | 63.20% | 33.33% | 0.7596 | 0.6900 |
| 6 | **step6_augmented** | 63.20% | 22.67% | 0.8229 | 0.7149 |

## 2. Table 3.3: Truncation vs. Chunking (Hypothesis H1)
| Variant | Start Recall | Middle Recall | End Recall |
| :--- | :---: | :---: | :---: |
| **step1_baseline** | 50.00% | 76.67% | 60.38% |
| **step2_normalizer** | 40.48% | 40.00% | 28.30% |
| **step3_chunking** | 26.19% | 26.67% | 30.19% |
| **step4_pair_input** | 35.71% | 30.00% | 37.74% |
| **step5_token_head** | 52.38% | 40.00% | 50.94% |
| **step6_augmented** | 59.52% | 66.67% | 60.38% |

## 3. Table 3.4: Recall by Attack Category (Hypothesis H3)
| Variant | Task-Irrelevant | Task-Relevant | Targeted |
| :--- | :---: | :---: | :---: |
| **step1_baseline** | 44.00% | 45.00% | 70.00% |
| **step2_normalizer** | 40.00% | 35.00% | 33.75% |
| **step3_chunking** | 32.00% | 25.00% | 27.50% |
| **step4_pair_input** | 40.00% | 30.00% | 35.00% |
| **step5_token_head** | 24.00% | 25.00% | 62.50% |
| **step6_augmented** | 40.00% | 30.00% | 76.25% |

## 4. Table 3.5: Obfuscation Robustness Suite (Hypothesis H2)
| Obfuscation Type | Baseline Recall (Step 1) | With Normalizer (Step 2) | With Normalizer & Augmentation (Step 6) |
| :--- | :---: | :---: | :---: |
| **Base64 wrapper** | 40.00% | 95.20% | 98.40% |
| **Homoglyphs** | 63.20% | 67.20% | 97.60% |
| **Zero-width characters** | 44.80% | 96.00% | 98.40% |
| **Leetspeak** | 63.20% | 96.00% | 99.20% |
| **Markdown embeds** | 58.40% | 96.00% | 98.40% |

*[... truncated at 60 lines — see full file: reports/ablation_summary.md]*

---

## 4d. Obfuscation Held-Out Results (team-reported, P3)

> Source: `results/table_3_5_obfuscation_comparison.json` (owner P3, please review)
> Provenance: team-reported (P3). n=206 for each obfuscation type.

| Obfuscation Type | Step1 Recall | Step2 Recall | Step6 Recall |
|-----------------|-------------|-------------|-------------|
| Base64 wrapper | 0.4000 | 0.9520 | 0.9840 |
| Homoglyphs | 0.6320 | 0.6720 | 0.9760 |
| Zero-width characters | 0.4480 | 0.9600 | 0.9840 |
| Leetspeak | 0.6320 | 0.9600 | 0.9920 |
| Markdown embeds | 0.5840 | 0.9600 | 0.9840 |

---

## 4e. Latency Per Stage incl. Model Forward (team-reported, P3)

> Source: `results/table_3_8_latency.json` (owner P3, please review)

> Hardware: cpu, ~2000-token document (team-reported P3)

| Stage | Mean (ms) | Median (ms) | p95 (ms) |
|-------|-----------|------------|---------|
| Keyword baseline | 43.56 | 41.17 | 55.14 |
| Full pipeline (model forward) | 166.61 | 167.83 | 185.01 |

Latency increase factor (full/baseline): 3.82×

---

## 5. P5 Evaluation

**Split**: test (N=206: 125 injected, 81 clean)
**Hard-neg**: N=15 (indicative only — wide CIs)

### 5a. Main Results Table

| Predictor | Recall | FPR (test) | FPR (hard-neg) | PSR | ROUGE-1 F1 |
|-----------|--------|------------|----------------|-----|------------|
| none | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.8962 |
| oracle | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.8996 |
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
3. **ASR proxy (PSR)** — PSR = payload survival rate. Real ASR requires LLM execution and
   behavioural verification. No LLM endpoint available; PSR is the proxy label.
4. **ROUGE-1 utility proxy** — surface-text overlap; actual utility requires downstream eval.
5. **token_head: N/A (weights absent)** — no trained weights committed; all token-head
   span-F1, PSR, ROUGE-1 metrics are N/A. Re-eval requires P4 to push weights.
6. **Sensitivity is SIMULATED** — gold span perturbations, not predicted spans from trained model.
7. **Obfuscated set sourced from table_3_5** — team-reported (P3). No separate held-out JSONL.
8. **pair-input eval present (step4)** — step4_pair_input metrics available (team-reported P3).
9. **No LLM ASR** — LLMBackend protocol stubbed; requires real LLM endpoint.
10. **Team baseline FPR (val) ~0.68** — honest estimate from baseline_day2.json (val split).

### Open Items

| Item | Status | Owner | Note |
|------|--------|-------|------|
| Trained token-head weights | BLOCKED | P4 | Needed for token_head, span-F1, real PSR |
| Real LLM ASR | N/A | All | No LLM endpoint; PSR proxy used |
| Full splits re-eval | OPEN | P1/P2 | team baseline n=42999 vs P5 n=206 |
| token_head re-eval | BLOCKED | P4 | weights absent in repo |

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
