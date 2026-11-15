# P5 Evaluation Report — Sudarshana Indirect Prompt-Injection Detection

## 1. Setup & Metric Definitions

**Dataset**: `data/splits/test.jsonl` (test split, seed=42)
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

0.9964 recall / 0.4456 FPR  (source: `baseline_day2.json`)

P5 lexical re-evaluation on `test` split:
- Recall: 0.0400
- FPR (test): 0.0123
- FPR (hard-neg): 0.0000

---

## 3. Recall by Injection Position

![Recall by Injection Position](/home/kinshuk-gupta/KINSHUK GUPTA/IIT BHILAI/IIT BHILAI 7TH SEMESTER/DSL504 - NATURAL LANGUAGE PROCESSING/Group Project/Sudarshana/reports/figures/recall_by_position.png)
*Recall by Injection Position*

| Predictor | Position | Support | Recall |
|-----------|----------|---------|--------|
| none | end | 53 | 0.0000 |
| none | middle | 30 | 0.0000 |
| none | start | 42 | 0.0000 |
| oracle | end | 53 | 1.0000 |
| oracle | middle | 30 | 1.0000 |
| oracle | start | 42 | 1.0000 |
| baseline | end | 53 | 0.0377 |
| baseline | middle | 30 | 0.0333 |
| baseline | start | 42 | 0.0476 |


---

## 4. Recall by Attack Category

![Recall by Attack Type](/home/kinshuk-gupta/KINSHUK GUPTA/IIT BHILAI/IIT BHILAI 7TH SEMESTER/DSL504 - NATURAL LANGUAGE PROCESSING/Group Project/Sudarshana/reports/figures/recall_by_attack_type.png)
*Recall by Attack Type*

| Predictor | Attack Type | Support | Recall |
|-----------|-------------|---------|--------|
| none | Base Encoding | 5 | 0.0000 |
| none | Blocking Internet Connection | 5 | 0.0000 |
| none | Bringing Down Hosts and Servers (Denial of Service) | 5 | 0.0000 |
| none | Business Intelligence | 5 | 0.0000 |
| none | Compromising Computers | 5 | 0.0000 |
| none | Conversational Agent | 5 | 0.0000 |
| none | Corrupting an Operating System | 5 | 0.0000 |
| none | Data Eavesdropping | 5 | 0.0000 |
| none | Emoji Substitution | 5 | 0.0000 |
| none | Encrypting Documents and Demanding Ransom (Ransomware) | 5 | 0.0000 |
| none | Entertainment | 5 | 0.0000 |
| none | Information Dissemination | 5 | 0.0000 |
| none | Introduce System Fingerprinting | 5 | 0.0000 |
| none | Keylogging | 5 | 0.0000 |
| none | Language Translation (Test) | 5 | 0.0000 |
| none | Marketing & Advertising | 5 | 0.0000 |
| none | Misinformation & Propaganda | 5 | 0.0000 |
| none | Research Assistance | 5 | 0.0000 |
| none | Reverse Text | 5 | 0.0000 |
| none | Scams & Fraud | 5 | 0.0000 |
| none | Screen Scraping | 5 | 0.0000 |
| none | Sentiment Analysis | 5 | 0.0000 |
| none | Substitution Ciphers | 5 | 0.0000 |
| none | Task Automation | 5 | 0.0000 |
| none | Traffic Analysis | 5 | 0.0000 |
| oracle | Base Encoding | 5 | 1.0000 |
| oracle | Blocking Internet Connection | 5 | 1.0000 |
| oracle | Bringing Down Hosts and Servers (Denial of Service) | 5 | 1.0000 |
| oracle | Business Intelligence | 5 | 1.0000 |
| oracle | Compromising Computers | 5 | 1.0000 |
| oracle | Conversational Agent | 5 | 1.0000 |
| oracle | Corrupting an Operating System | 5 | 1.0000 |
| oracle | Data Eavesdropping | 5 | 1.0000 |
| oracle | Emoji Substitution | 5 | 1.0000 |
| oracle | Encrypting Documents and Demanding Ransom (Ransomware) | 5 | 1.0000 |
| oracle | Entertainment | 5 | 1.0000 |
| oracle | Information Dissemination | 5 | 1.0000 |
| oracle | Introduce System Fingerprinting | 5 | 1.0000 |
| oracle | Keylogging | 5 | 1.0000 |
| oracle | Language Translation (Test) | 5 | 1.0000 |
| oracle | Marketing & Advertising | 5 | 1.0000 |
| oracle | Misinformation & Propaganda | 5 | 1.0000 |
| oracle | Research Assistance | 5 | 1.0000 |
| oracle | Reverse Text | 5 | 1.0000 |
| oracle | Scams & Fraud | 5 | 1.0000 |
| oracle | Screen Scraping | 5 | 1.0000 |
| oracle | Sentiment Analysis | 5 | 1.0000 |
| oracle | Substitution Ciphers | 5 | 1.0000 |
| oracle | Task Automation | 5 | 1.0000 |
| oracle | Traffic Analysis | 5 | 1.0000 |
| baseline | Base Encoding | 5 | 0.0000 |
| baseline | Blocking Internet Connection | 5 | 0.0000 |
| baseline | Bringing Down Hosts and Servers (Denial of Service) | 5 | 0.0000 |
| baseline | Business Intelligence | 5 | 0.0000 |
| baseline | Compromising Computers | 5 | 0.0000 |
| baseline | Conversational Agent | 5 | 0.0000 |
| baseline | Corrupting an Operating System | 5 | 0.4000 |
| baseline | Data Eavesdropping | 5 | 0.0000 |
| baseline | Emoji Substitution | 5 | 0.0000 |
| baseline | Encrypting Documents and Demanding Ransom (Ransomware) | 5 | 0.2000 |
| baseline | Entertainment | 5 | 0.0000 |
| baseline | Information Dissemination | 5 | 0.0000 |
| baseline | Introduce System Fingerprinting | 5 | 0.2000 |
| baseline | Keylogging | 5 | 0.0000 |
| baseline | Language Translation (Test) | 5 | 0.0000 |
| baseline | Marketing & Advertising | 5 | 0.0000 |
| baseline | Misinformation & Propaganda | 5 | 0.0000 |
| baseline | Research Assistance | 5 | 0.0000 |
| baseline | Reverse Text | 5 | 0.0000 |
| baseline | Scams & Fraud | 5 | 0.0000 |
| baseline | Screen Scraping | 5 | 0.2000 |
| baseline | Sentiment Analysis | 5 | 0.0000 |
| baseline | Substitution Ciphers | 5 | 0.0000 |
| baseline | Task Automation | 5 | 0.0000 |
| baseline | Traffic Analysis | 5 | 0.0000 |


---

## 5. FPR on Hard Negatives

| Predictor | FPR (test negatives) | FPR (hard negatives) |
|-----------|----------------------|----------------------|
| none | 0.0000 | 0.0000 |
| oracle | 0.0000 | 0.0000 |
| baseline | 0.0123 | 0.0000 |


---

## 6. Downstream: PSR and ROUGE-1 (none vs oracle vs predicted)

![PSR and ROUGE-1 by Predictor](/home/kinshuk-gupta/KINSHUK GUPTA/IIT BHILAI/IIT BHILAI 7TH SEMESTER/DSL504 - NATURAL LANGUAGE PROCESSING/Group Project/Sudarshana/reports/figures/psr_rouge_by_condition.png)
*PSR and ROUGE-1 by Predictor*

| Predictor | PSR (↓ better for defence) | ROUGE-1 F1 (↑ better) |
|-----------|---------------------------|----------------------|
| none | 1.0000 | 0.8962 |
| oracle | 0.0000 | 0.7298 |
| baseline | 1.0000 | 0.8967 |


**PSR interpretation**: Lower PSR = defence removed more attack payload.
**ROUGE-1 interpretation**: Higher F1 = more of the clean content preserved.

---

## 7. Latency

![Pipeline Latency p50/p95](/home/kinshuk-gupta/KINSHUK GUPTA/IIT BHILAI/IIT BHILAI 7TH SEMESTER/DSL504 - NATURAL LANGUAGE PROCESSING/Group Project/Sudarshana/reports/figures/latency_p50_p95.png)
*Pipeline Latency p50/p95*

Benchmark: ~1783 estimated tokens (12674 chars). 20 runs, 3 warmups.

| Stage | p50 (ms) | p95 (ms) | Mean (ms) |
|-------|----------|----------|-----------|
| normalize_pysbd | 31.90 | 50.41 | 33.54 |
| redact | 16.53 | 25.80 | 18.12 |
| tokenizer_encode | 2.18 | 3.51 | 2.40 |
| model_forward | N/A | N/A | N/A (no weights — checkpoints/ contain only config+tokenizer files) |


---

## 8. Obfuscated Held-Out Status

> **SKIPPED**: no obfuscated set yet; P1/P2 deliverable

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
