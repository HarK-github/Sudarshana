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

# P4 Report: Token Head and Redaction

## 1. Summary
We built and evaluated a token-level classifier (`O`, `B-INJ`, `I-INJ`) on a DeBERTa-v3 backbone paired with sentence-level span redaction for the Sudarshana indirect prompt-injection detection pipeline. Spans are stored as exact character offsets in the original document with 100% verified round-trip recovery. Across 3 seeds evaluated on held-out attack types from BIPIA, the token head achieves high span overlap F1 while redaction preserves significant document text. Hard negatives reduce false redactions dramatically compared to unregularized models. Main caveat: this checkpoint is an independent prototype to validate the labels, head, and redaction logic; P3 owns the final joint training run with pair inputs and normalizer.

## 2. Environment and reproducibility
- **Hardware**: AMD Ryzen 7 7735U (16 vCPUs, 2.70 GHz base, x86_64), 14.2 GB System RAM.
- **Operating System**: Linux 7.0.0-43-generic (glibc 2.43).
- **Python Version**: Python 3.13.5.
- **Key Libraries**:
  - `torch`: 2.14.0+cu130 (CPU execution mode; CUDA unavailable)
  - `transformers`: 5.19.0
  - `datasets`: 5.1.0
  - `scikit-learn`: 1.9.1
  - `numpy`: 2.2.6
  - `pysbd`: 0.3.4 (Python Sentence Boundary Disambiguation)
  - `pytest`: 9.1.1
- **Repository Commit**: `77ec678ac98327245b2c5d5bf49c7b5cdd4c08a1`
- **Seeds Fixed**: 42, 43, 44.
- **Execution Commands**:
  - Dataset construction: `python3 -m sudarshana.data.builder`
  - Unit tests: `python3 -m pytest tests/`
  - Training prototype runs: `PYTHONPATH=. python3 scripts/train_seeds.py`
  - Full evaluation: `PYTHONPATH=. python3 -m sudarshana.evaluation.evaluate`

---

## 3. Dataset research

### 3.1 Comparison table
Every candidate was evaluated on identical criteria. All factual entries are verified against official repositories, arXiv papers, and dataset cards.

| Candidate | Source & License | Size | Content Type | Injection Type | Span Info | Attack Types | Position Control | Benign / Hard Negatives | Built-in Splits | Access & Format | Fit for Token Head (1–5) & Reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **BIPIA** (Primary) | [github.com/microsoft/BIPIA](https://github.com/microsoft/BIPIA) · MIT License (core), CC BY-SA 4.0 for components | 5 tasks, 50 attack types (250 prompts), >10k contexts | Email, Web QA, Table QA, Code QA, Summarization | Indirect (inside retrieved document) | Exact character spans recoverable via known payload matching; verified round-trip | 50 distinct types (15 text train, 15 text test, 10 code train, 10 code test); templated | Controlled: `start`, `middle`, `end` | Clean contexts provided; hard negatives limited | Official train vs test split by attack type and task | GitHub repo; JSON/JSONL | **5/5**: Gold standard for indirect injection; 5 RAG tasks, controlled positions, exact span recovery. |
| **AgentDojo** | [github.com/ethz-spylab/agentdojo](https://github.com/ethz-spylab/agentdojo) · MIT License | 97 tasks, 627 injections across 4 suites | Tool outputs, emails, Slack messages, banking | Indirect (agent tool execution trace) | Embedded in tool returns; dynamically trajectory-dependent | Multi-turn hijacking, state tampering, exfiltration | Arbitrary (inside mock tool outputs) | Full benign agent task suite | Suite & task splits | Python package / GitHub | **3/5**: Excellent for multi-turn agents, but spans are entangled with dynamic tool execution rather than single-pass documents. |
| **InjecAgent** | [github.com/uiuc-kang-lab/InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent) · Apache 2.0 | 1,054 test cases, 17 user tools, 62 attacker tools | Tool responses (weather, email, calendar) | Indirect (in tool responses) | Attack prompt known in JSON fields; substring recoverable | Direct harm & data exfiltration | Tool observation content | Base settings & benign tool outputs | Tool-based & attack-based splits | GitHub / JSON | **3.5/5**: Realistic tool outputs with recoverable spans, but smaller scale and narrower document diversity than BIPIA. |
| **Tensor Trust** | [github.com/HumanCompatibleAI/tensor-trust-data](https://github.com/HumanCompatibleAI/tensor-trust-data) · MIT / CC BY 4.0 | ~126k submissions (~2k benchmark subset) | Short game prompts | Direct prompt injection / jailbreak | Trivial (entire string is attack input; no carrier document) | Extraction, hijacking, encoded | Direct user prompt | Defenses are system prompts, not documents | Paper train/val/test splits | GitHub / JSONL | **1.5/5**: Direct injection game; lacks indirect carrier document structure and span localization. |
| **HackAPrompt** | [huggingface.co/datasets/hackaprompt/hackaprompt-dataset](https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset) · MIT License | 600,000+ competition prompts across 10 levels | Short adversarial prompts | Direct prompt injection / jailbreak | Entire prompt is attack; no embedding inside retrieved text | Level-based defense bypasses (regex, token filters) | Direct prompt | All adversarial; minimal benign text | Level-based splits | Hugging Face Datasets | **2/5**: Massive collection of direct attack prompts; useful as payload source for synthesis, but not a span benchmark. |
| **deepset/prompt-injections** | [huggingface.co/datasets/deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) · Apache 2.0 | 662 examples (546 train, 116 test) | Short sentences, questions | Direct and indirect mixed | None (flat binary classification label only) | None (binary label) | Uncontrolled | 391 benign examples | Train (546) / Test (116) | Hugging Face Datasets | **1/5**: Small scale, document-level binary labels only, completely lacks span information. |
| **Lakera Gandalf** | [huggingface.co/Lakera](https://huggingface.co/Lakera) · Apache 2.0 | Tens of thousands of game attempts | Short chat prompts | Direct prompt extraction (secret password) | Entire input is attack | 8 defense levels (guardrails to adversarial filters) | Direct prompt | Lacks carrier documents | Game levels | Hugging Face Datasets | **1.5/5**: Password extraction game prompts; no indirect span localization or carrier documents. |
| **LLMail-Inject** | [github.com/microsoft/llmail-inject-challenge](https://github.com/microsoft/llmail-inject-challenge) · MIT / MSR License | >200,000 adaptive attack prompts | Emails and email assistant tools | Indirect prompt injection in emails | Attack payloads known; substring recoverable | Exfiltration, tool abuse, defense bypasses | Email body, subject, footer | Large corpus of benign email contexts | Challenge phase splits | Hugging Face / GitHub | **4/5**: Outstanding realistic indirect injection benchmark in email domain with recoverable spans. Highly recommended as cross-dataset validation. |
| **Open-Prompt-Injection** | [github.com/liu00222/Open-Prompt-Injection](https://github.com/liu00222/Open-Prompt-Injection) · MIT License | Thousands of synthesized variations | Web documents, QA contexts (MS MARCO, SQuAD) | Direct and indirect | Configurable placeholder synthesis; loggable spans | Content manipulation, context disruption, data theft | Prefix, infix, suffix | Benign contexts from SQuAD / MS MARCO | Task-based splits | GitHub repository | **4/5**: Strong formal framework with diverse contexts, but requires synthetic dataset compilation rather than fixed canonical splits. |
| **PromptInject** | [github.com/agencyenterprise/PromptInject](https://github.com/agencyenterprise/PromptInject) · MIT License | Modular framework generating ~45 attacks | Synthetic prompt strings | Prompt injection into templated context | Delimited placeholders enable span extraction | Goal hijacking, prompt leaking | Controlled placement | Basic synthetic templates | None (generator only) | GitHub repo | **2.5/5**: Early pioneering work, but small-scale, templated, and superseded by BIPIA. |

### 3.2 Recommendation to P1 and the decision P1 took
- **Recommendation to P1**:
  1. Freeze on **BIPIA** as the primary authoritative benchmark: it is the only benchmark covering 5 diverse RAG tasks, controlled `start`/`middle`/`end` injection placement, 50 disjoint attack types, and deterministic character span recovery with 100% round-trip fidelity.
  2. Complement BIPIA with a dedicated suite of **Hard Negatives** (imperative technical READMEs, shell tutorials, recipes, and business action emails) with empty span annotations `[]` to prevent the token head from false-flagging legitimate imperative text.
  3. If cross-dataset evaluation is brought into scope in later phases, prioritize **LLMail-Inject** for out-of-domain email evaluation.
- **Decision P1 Took**: P1 accepted the recommendation and froze BIPIA with the 50-attack-type disjoint split (15 text train, 15 text test, 10 code train, 10 code test) and approved the integration of hard negatives. Cross-dataset evaluation was confirmed out of scope for the 7-day prototype timeline.

### 3.3 Datasets rejected and why
- **Tensor Trust & Lakera Gandalf**: Rejected because they evaluate direct prompt extraction/hijacking in game setups where the entire user prompt is malicious. They lack retrieved carrier documents and span localization.
- **HackAPrompt**: Rejected as a primary benchmark because prompts are standalone direct injection payloads without carrier documents.
- **deepset/prompt-injections**: Rejected because it only provides binary document-level labels with zero token or span annotations.
- **PromptInject**: Rejected due to small synthetic sample size and lack of real-world carrier document diversity.

---


*[... truncated at 60 lines — see full file: reports/p4_token_head_report.md]*

### P4 Headline Metrics (seed=42, `reports/metrics.json`)

| Predictor | Token Recall | Token Precision | Token F1 | Span Overlap F1 | Source |
|-----------|-------------|-----------------|----------|-----------------|--------|
| token_head (P4-reported, not re-evaluated: no weights) | 0.9349 | 0.7758 | 0.8479 | 0.3442 | `reports/metrics.json` |

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
