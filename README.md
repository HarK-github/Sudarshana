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
           ▼
retrieved document ──► [ SUDARSHANA ] ──► clean document ──► any LLM (GPT-4, Claude, local, ...)
                         normalize → chunk → detect → redact
```

The generator never sees an unredacted document. No changes to the LLM. No weight access required.

---

## ⚙️ Features in use

### Preprocessing & length
- **Normalizer**: NFKC, zero-width character stripping, base64/hex decoding before tokenization; offset map back to original text.
- **Sliding-window chunker**: 256-token chunks with 64-token overlap.

### Model
- **Pair input format**: `[CLS] user task [SEP] chunk [SEP]`.
- **Shared DeBERTa-v3 encoder** for all heads.
- **Chunk-level detection head**: one malicious/safe score per chunk.
- **Max / top-k pooling** over chunk scores.
- **Token-level head**: `O` / `B-INJ` / `I-INJ`.
- **Optional attack-type head** for regularization and logging (only if Day 5 has slack).

### Decision & action
- **Calibrated threshold** at a target false-positive rate.
- **Sentence-level redaction** of flagged spans, with **block fallback** for high flagged share or low confidence.
- **Joint training loss**: detection loss + token loss + optional attack-type loss.

### Data & evaluation
- **Obfuscation augmentation** on training data only.
- Attack-type split, hard negatives, counterfactual pairs, span labels, obfuscated held-out set, separate code-QA reporting.
- **Metrics**: recall/FPR, recall by position and attack category, span F1, end-to-end ASR, ROUGE-1 on clean/redacted documents, latency per ~2k-token document.

---

## 📈 What it improves over the baseline

| Baseline weakness | Sudarshana fix |
|---|---|
| Injection hidden after token 512 never seen | Sliding window + max pooling — clearest win |
| Base64, homoglyphs, zero-width, leetspeak bypass tokenizer | Normalizer + obfuscation augmentation |
| Imperative text in recipes/READMEs/tutorials gets flagged | `(user task, chunk)` pair input lowers false positives |
| Whole document blocked when one injection is found | Token head + sentence-level redaction |
| No hint of attack type | Optional attack-type head |
| Threshold picked arbitrarily | Calibration at a target FPR |

> ⚠️ **No measured results yet.** Every expected effect above is a hypothesis to be checked against the Day-1 baseline. Baseline percentages are a Day-1 output, not a known number.

---

## 🚫 What it deliberately does not do

- **Split-payload attacks.** An instruction spread across distant chunks can slip past max pooling.
- **Semantic attacks with no imperative phrasing.** Trained on templated data; can miss paraphrased attacks.
- **Adaptive attackers.** Anyone who knows the detector can craft inputs against it. **Sudarshana is one defense layer, not a complete one.**
- **Replace BIPIA defenses.** The paper's white-box result (ASR near zero) is strong; combining Sudarshana with the explicit-reminder prompt is cheap and sensible.

---

## 🏗️ Pipeline architecture

```text
raw document
  → Normalizer (NFKC, strip zero-width, decode base64/hex)      [code]
  → Chunker (256 tokens, 64 overlap)                            [code]
  → DeBERTa-v3 encoder, input = (user task, chunk)              [shared backbone]
      ├─ chunk-level head → malicious score
      └─ token-level head → injected spans (O / B-INJ / I-INJ)
  → Max / top-k pooling over chunk scores                       [code]
  → Calibrated threshold → block, or redact flagged spans       [code]
```

**Inference steps**

1. Normalize the document; keep an offset map back to the original text.
2. Split into 256-token chunks with 64-token overlap; prefix each with the user task.
3. Score each chunk; take max or top-k over chunk scores.
4. Compare pooled score with threshold calibrated to a target false-positive rate.
5. If it fires, use the token head's spans to redact.

**Redaction rules**

- Expand each flagged token to its **whole sentence** — a missed token must not leave a residual payload.
- Map span offsets back to the original text before cutting (normalization shifts offsets).
- Tune the redaction threshold for **recall**, not F1.
- **Fall back to blocking** the whole document when the flagged share is large or confidence is low.
- Verify the redacted document still supports the user's task.

**Joint loss**

$$
\mathcal{L} = \mathcal{L}_{\text{detect}} + \lambda_{\text{tok}}\,\mathcal{L}_{\text{tok}} + \lambda_{\text{type}}\,\mathcal{L}_{\text{type}}
$$

Weights tuned on validation data; no fixed values.

---

## 📁 Repository layout

```text
sudarshana/
├── data/                  # unified dataset builders, splits, hard negatives
├── preprocessing/         # normalizer, offset map, chunker, augmentation
├── models/                # DeBERTa wrapper, chunk head, token head, attack-type head
├── training/              # collators, training loop, loss composition
├── inference/             # pipeline: normalize → chunk → score → pool → redact
├── redaction/             # sentence expansion, offset mapping, block fallback
├── evaluation/            # recall/FPR, span F1, ASR, ROUGE-1, latency
├── configs/               # ablation configs and calibration targets
├── scripts/               # reproduce baseline, ablations, eval tables
└── docs/                  # design notes, model card, results
```

---

## 🚀 Getting started

```bash
# 1. Clone
git clone <repo-url> sudarshana && cd sudarshana

# 2. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Build the unified dataset and split
python -m sudarshana.data.build --config configs/data.yaml

# 4. Train (pair input, joint heads)
python -m sudarshana.training.train --config configs/train.yaml

# 5. Evaluate (ablation ladder)
python -m sudarshana.evaluation.run --config configs/eval.yaml --ladder full
```

---

## 👥 Team & ownership

<div align="center">

| Person | Role | Features owned |
|---|---|---|
| **P1** | Data & Baseline Lead | Dataset schema, attack-type/source split, hard negatives, counterfactual pairs, baseline run, threshold calibration, ablation coordination |
| **P2** | Preprocessing & Chunking | Normalizer, offset map, sliding-window chunker, max/top-k pooling, obfuscation augmentation generator |
| **P3** | Model & Training Integration | Shared DeBERTa-v3 encoder, pair input format, joint training loss, optional attack-type head, retraining, packaging |
| **P4** | Token Head & Redaction | Token-level labels, token head, span F1, sentence-level redaction, block fallback, residual-payload checks |
| **P5** | Downstream Evaluation & Reporting | End-to-end ASR, ROUGE-1, latency, obfuscated held-out evaluation, metrics tables, write-up |

</div>

---

## 🗓️ 7-day plan

> 5 full-time people · shared repo · daily 15-minute sync · **Day 7 is buffer/write-up only**.

<div align="center">

| Day | P1 Data/Baseline | P2 Preprocess/Chunk | P3 Model/Training | P4 Token/Redaction | P5 Eval/Reporting |
|---|---|---|---|---|---|
| **1** | Unify BIPIA/AgentDojo/TensorTrust into `(user_task, document, label, injection_span, attack_type)`; split by attack type and source; baseline harness | Normalizer + unit tests; offset-map design | Reproduce baseline training; training-harness skeleton | Span-label spec from insertion points; token-head scaffold | Downstream eval harness: fixed-LLM ASR, ROUGE-1, latency; baseline numbers |
| **2** | Hard negatives; record baseline metrics; freeze data contract | Chunker 256/64 + max/top-k pooling; integrate normalizer + chunker into inference | Pair-input collator `[CLS] task [SEP] chunk [SEP]`; training loop | Token-head implementation; span-label generation script | Baseline report; recall-by-position and recall-by-attack-category harness |
| **3** | Counterfactual pairs; FPR-at-fixed-recall eval | Offset map through normalize/decode/chunk; tests | Train pair input; compare vs chunk-only; recall by attack category | Token-head joint-training scaffold; span-F1 metric | FPR on hard negatives; pair-input eval table |
| **4** | Span-label QA with P4; validation split | Redaction offset integration support | Joint detection + token loss; integrate both heads | Sentence-level redaction + block fallback; redaction checks | ASR / ROUGE after redaction; downstream-task check |
| **5** | Obfuscated held-out set; eval protocol | Obfuscation augmentation generator, training data only | Retrain with augmentation; attack-type head only if time | Residual-payload tests; span F1 on held-out | Obfuscated held-out eval; latency per ~2k-token document |
| **6** | Calibrate threshold at target FPR; coordinate full ablation ladder | Preprocessing / latency ablations; fixes | Run ablation variants; **freeze model** | Tune redaction threshold for recall; validate fallback | Full metrics table: recall/FPR, span F1, ASR, ROUGE-1, latency |
| **7** | Results, limitations, open items | Pipeline packaging | Final model card / integration docs | Redaction documentation | Final report and figures |

</div>

### Ablation ladder

1. Baseline — truncated, single string
2. + Normalizer
3. + Chunking with max pooling
4. + Pair input
5. + Token head and redaction
6. + Obfuscation augmentation

Each layer is added on top of the previous one so the gain of each is visible.

---

## 🔗 Critical dependencies

- **Day 1 data contract from P1 blocks everyone.** P3 cannot train fair variants and P5 cannot evaluate without the split.
- **P3 owns the shared training harness.** P2 and P4 integrate into it; no duplicate training code.
- **P4 depends on P2's offset map** for redaction after normalization/decoding.
- **P5 depends on P1's splits and P3/P4 artifacts**; P5 builds the harness on Day 1 rather than wait.
- **Day 6 is the freeze point.** Day 7 adds no features.

---

## 📊 Metrics we report

<div align="center">

| Metric | Why |
|---|---|
| Recall and FPR on held-out attack types | Main detector result |
| Recall by position (start / middle / end) | Shows the 512-token truncation problem |
| FPR on hard negatives at fixed recall | Cost of ignoring the user task |
| Recall per attack category (task-irrelevant / task-relevant / targeted) | Pair input may trade recall for precision |
| Recall on obfuscated held-out attacks | Value of normalizer and augmentation |
| Span-level F1 | Quality of the token head |
| End-to-end ASR with and without the filter | Comparable to BIPIA's table |
| ROUGE-1 on BIPIA-Clean | Redaction must not hurt normal tasks |
| Latency per ~2k-token document | Cost |

</div>

**Reference point (BIPIA paper).** No-defense ASR: 31.03% (GPT-4), 26.16% (GPT-3.5-turbo), 15.31% (Vicuna-13B), 12.37% (Vicuna-7B) overall. Paper defenses: ~20.6% (GPT-4, multi-turn dialogue) and ~0.5–2.4% (Vicuna-7B, white-box fine-tuning). These are **generator** results, not detector accuracy; our end-to-end ASR is the comparable figure.

---

## ⚠️ Known limitations

- **Split-payload attacks** across distant chunks can slip past max pooling.
- **Semantic attacks** with no imperative phrasing can still be missed.
- **Adaptive attackers** who know the detector can craft inputs against it. One layer, not a complete defense.
- **Templated data.** BIPIA has only five instructions per attack type; held-out attack types are the main check against phrasing overfit.
- **Code QA is different.** Code attacks are snippets, not natural-language imperatives; report text and code separately.
- **Pair input risk.** BIPIA found task-relevant and targeted attacks succeed more often; a model learning "does this derail the user's goal?" can miss exactly those. Watch recall per attack category.

---

## 📜 License

TBD — to be decided before public release.

---

<div align="center">

*Sudarshana — harden the pipeline, not just the prompt.*

</div>
