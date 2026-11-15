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
| keyword_heuristic | P5 lexical keyword scorer (threshold=0.5, document-only). NOT the team baseline. |
| token_head | N/A (no weights — checkpoints contain only config+tokenizer files) |

### ⚠️ Proxy Metric Warnings

**PSR ("ASR proxy — payload survival rate")** is NOT real Attack Success Rate.
Real ASR requires an LLM to execute the injected instruction and confirm behavioural change.
PSR only measures whether the injection *text* survives in the post-redaction document.

**ROUGE-1 F1 ("ROUGE-1 utility proxy")** is NOT real downstream task utility.
Real utility requires running the downstream agent/LLM on the redacted document and
comparing task completion. ROUGE-1 is a surface-text overlap proxy only.

---

## 2. Team Baseline (lexical model, `baseline_day2.json` — teammate result, not recomputed)

> **Ran on full dataset** (train=179 820 / val=44 729 / test=42 999). Committed sample splits
> (test=206, hard_neg=15) are **not comparable**.

| Split | N | Recall | FPR | Precision | F1 |
|-------|---|--------|-----|-----------|-----|
| val | 44729 | 0.9982 | 0.6780 | 0.9951 | 0.9967 |
| test | 42999 | 0.9964 | 0.4456 | 0.9890 | 0.9927 |

**Recall by attack type (val split)**

| Attack Type | Support | Recall |
|-------------|---------|--------|
| Cryptocurrency Mining | 750 | 0.9800 |
| Device and Driver Enumeration | 750 | 1.0000 |
| Malware Distribution | 14250 | 1.0000 |
| Programming Help | 14250 | 1.0000 |
| Space Removal & Grouping | 14250 | 1.0000 |
| injection_task_0 | 40 | 0.0250 |
| prompt_hijack | 116 | 0.7931 |

### P5 keyword_heuristic re-evaluation on `test` split (N=125)

- Recall: 0.0400 [0.000, 1.000] (n=0)
- FPR (test): 0.0123 [0.000, 1.000] (n=0)
- FPR (hard-neg N=15, indicative only): 0.0000

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
| keyword_heuristic | end | 53 | 0.0377 |
| keyword_heuristic | middle | 30 | 0.0333 |
| keyword_heuristic | start | 42 | 0.0476 |


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
| keyword_heuristic | Base Encoding | 5 | 0.0000 |
| keyword_heuristic | Blocking Internet Connection | 5 | 0.0000 |
| keyword_heuristic | Bringing Down Hosts and Servers (Denial of Service) | 5 | 0.0000 |
| keyword_heuristic | Business Intelligence | 5 | 0.0000 |
| keyword_heuristic | Compromising Computers | 5 | 0.0000 |
| keyword_heuristic | Conversational Agent | 5 | 0.0000 |
| keyword_heuristic | Corrupting an Operating System | 5 | 0.4000 |
| keyword_heuristic | Data Eavesdropping | 5 | 0.0000 |
| keyword_heuristic | Emoji Substitution | 5 | 0.0000 |
| keyword_heuristic | Encrypting Documents and Demanding Ransom (Ransomware) | 5 | 0.2000 |
| keyword_heuristic | Entertainment | 5 | 0.0000 |
| keyword_heuristic | Information Dissemination | 5 | 0.0000 |
| keyword_heuristic | Introduce System Fingerprinting | 5 | 0.2000 |
| keyword_heuristic | Keylogging | 5 | 0.0000 |
| keyword_heuristic | Language Translation (Test) | 5 | 0.0000 |
| keyword_heuristic | Marketing & Advertising | 5 | 0.0000 |
| keyword_heuristic | Misinformation & Propaganda | 5 | 0.0000 |
| keyword_heuristic | Research Assistance | 5 | 0.0000 |
| keyword_heuristic | Reverse Text | 5 | 0.0000 |
| keyword_heuristic | Scams & Fraud | 5 | 0.0000 |
| keyword_heuristic | Screen Scraping | 5 | 0.2000 |
| keyword_heuristic | Sentiment Analysis | 5 | 0.0000 |
| keyword_heuristic | Substitution Ciphers | 5 | 0.0000 |
| keyword_heuristic | Task Automation | 5 | 0.0000 |
| keyword_heuristic | Traffic Analysis | 5 | 0.0000 |


---

## 5. FPR on Hard Negatives

| Predictor | FPR (test negatives) | FPR (hard negatives) |
|-----------|----------------------|----------------------|
| none | 0.0000 | 0.0000 |
| oracle | 0.0000 | 0.0000 |
| keyword_heuristic | 0.0123 | 0.0000 |


---

## 6. Downstream: PSR and ROUGE-1 (none vs oracle vs predicted)

![PSR and ROUGE-1 by Predictor](/home/kinshuk-gupta/KINSHUK GUPTA/IIT BHILAI/IIT BHILAI 7TH SEMESTER/DSL504 - NATURAL LANGUAGE PROCESSING/Group Project/Sudarshana/reports/figures/psr_rouge_by_condition.png)
*PSR and ROUGE-1 by Predictor*

| Predictor | PSR (↓ better for defence) | ROUGE-1 F1 (↑ better) |
|-----------|---------------------------|----------------------|
| none | 1.0000 | 0.8962 |
| oracle | 0.0000 | 0.7298 |
| keyword_heuristic | 1.0000 | 0.8967 |


**PSR interpretation**: Lower PSR = defence removed more attack payload.
**ROUGE-1 interpretation**: Higher F1 = more of the clean content preserved.

---

## 7. Latency

![Pipeline Latency p50/p95](/home/kinshuk-gupta/KINSHUK GUPTA/IIT BHILAI/IIT BHILAI 7TH SEMESTER/DSL504 - NATURAL LANGUAGE PROCESSING/Group Project/Sudarshana/reports/figures/latency_p50_p95.png)
*Pipeline Latency p50/p95*

Benchmark: ~1783 estimated tokens (12674 chars). 20 runs, 3 warmups.

| Stage | p50 (ms) | p95 (ms) | Mean (ms) |
|-------|----------|----------|-----------|
| sentence_segmentation | 14.74 | 17.77 | 15.35 |
| redact | 14.57 | 14.68 | 14.57 |
| tokenizer_encode | 2.03 | 2.15 | 2.05 |
| model_forward | N/A | N/A | N/A (no weights — checkpoints/ contain only config+tokenizer files) |


---

## 8. Obfuscated Held-Out Status

> **SKIPPED**: no obfuscated set yet; P1/P2 deliverable

---

## 9. ⚠️ SIMULATED Redaction Sensitivity Analysis

> All results in this section are **SIMULATED**. Gold spans are perturbed to model imperfect
> redaction (partial span coverage, span drop). Real results require trained span predictions.

**Label**: SIMULATED redaction sensitivity — gold spans perturbed; not real model predictions

### 9a. PSR and ROUGE-1 Grid (keep_fraction f × drop_rate m)

| keep_fraction | drop_rate | PSR [95% CI] (n=injected) | ROUGE-1 F1 |
|---------------|-----------|---------------------------|------------|
| 1.0 | 0.0 | 0.0000 [0.000, 0.030] (n=125) | 0.7298 |
| 1.0 | 0.1 | 0.0000 [0.000, 0.030] (n=125) | 0.7298 |
| 1.0 | 0.25 | 0.0000 [0.000, 0.030] (n=125) | 0.7298 |
| 1.0 | 0.5 | 0.0000 [0.000, 0.030] (n=125) | 0.7298 |
| 0.9 | 0.0 | 0.0000 [0.000, 0.030] (n=125) | 0.7452 |
| 0.9 | 0.1 | 0.0000 [0.000, 0.030] (n=125) | 0.7452 |
| 0.9 | 0.25 | 0.0000 [0.000, 0.030] (n=125) | 0.7452 |
| 0.9 | 0.5 | 0.0000 [0.000, 0.030] (n=125) | 0.7452 |
| 0.75 | 0.0 | 0.0400 [0.017, 0.090] (n=125) | 0.7649 |
| 0.75 | 0.1 | 0.0400 [0.017, 0.090] (n=125) | 0.7649 |
| 0.75 | 0.25 | 0.0400 [0.017, 0.090] (n=125) | 0.7649 |
| 0.75 | 0.5 | 0.0400 [0.017, 0.090] (n=125) | 0.7649 |
| 0.5 | 0.0 | 0.2880 [0.216, 0.373] (n=125) | 0.8824 |
| 0.5 | 0.1 | 0.2880 [0.216, 0.373] (n=125) | 0.8824 |
| 0.5 | 0.25 | 0.2880 [0.216, 0.373] (n=125) | 0.8824 |
| 0.5 | 0.5 | 0.2880 [0.216, 0.373] (n=125) | 0.8824 |

### 9b. Oracle PSR and ROUGE-1 by Attack Type (f=1.0, m=0.0)

![Oracle PSR by Attack Type](/home/kinshuk-gupta/KINSHUK GUPTA/IIT BHILAI/IIT BHILAI 7TH SEMESTER/DSL504 - NATURAL LANGUAGE PROCESSING/Group Project/Sudarshana/reports/figures/psr_by_attack_type.png)
*Oracle PSR by Attack Type*

| Attack Type | N | PSR [95% CI] | ROUGE-1 F1 |
|-------------|---|--------------|------------|
| Base Encoding | 5 | 0.0000 [0.000, 0.434] | 0.9787 |
| Blocking Internet Connection | 5 | 0.0000 [0.000, 0.434] | 0.7917 |
| Bringing Down Hosts and Servers (Denial of Service) | 5 | 0.0000 [0.000, 0.434] | 0.5959 |
| Business Intelligence | 5 | 0.0000 [0.000, 0.434] | 0.7849 |
| Compromising Computers | 5 | 0.0000 [0.000, 0.434] | 0.7720 |
| Conversational Agent | 5 | 0.0000 [0.000, 0.434] | 0.9763 |
| Corrupting an Operating System | 5 | 0.0000 [0.000, 0.434] | 0.5870 |
| Data Eavesdropping | 5 | 0.0000 [0.000, 0.434] | 0.2000 |
| Emoji Substitution | 5 | 0.0000 [0.000, 0.434] | 0.9794 |
| Encrypting Documents and Demanding Ransom (Ransomware) | 5 | 0.0000 [0.000, 0.434] | 0.1979 |
| Entertainment | 5 | 0.0000 [0.000, 0.434] | 0.9970 |
| Information Dissemination | 5 | 0.0000 [0.000, 0.434] | 0.5888 |
| Introduce System Fingerprinting | 5 | 0.0000 [0.000, 0.434] | 0.3886 |
| Keylogging | 5 | 0.0000 [0.000, 0.434] | 0.4000 |
| Language Translation (Test) | 5 | 0.0000 [0.000, 0.434] | 0.9794 |
| Marketing & Advertising | 5 | 0.0000 [0.000, 0.434] | 0.9534 |
| Misinformation & Propaganda | 5 | 0.0000 [0.000, 0.434] | 0.9938 |
| Research Assistance | 5 | 0.0000 [0.000, 0.434] | 0.9913 |
| Reverse Text | 5 | 0.0000 [0.000, 0.434] | 0.9666 |
| Scams & Fraud | 5 | 0.0000 [0.000, 0.434] | 0.9773 |
| Screen Scraping | 5 | 0.0000 [0.000, 0.434] | 0.0000 |
| Sentiment Analysis | 5 | 0.0000 [0.000, 0.434] | 0.9645 |
| Substitution Ciphers | 5 | 0.0000 [0.000, 0.434] | 0.9888 |
| Task Automation | 5 | 0.0000 [0.000, 0.434] | 0.7922 |
| Traffic Analysis | 5 | 0.0000 [0.000, 0.434] | 0.3991 |

### 9c. Oracle PSR and ROUGE-1 by Position (f=1.0, m=0.0)

| Position | N | PSR [95% CI] | ROUGE-1 F1 |
|----------|---|--------------|------------|
| end | 53 | 0.0000 [0.000, 0.068] | 0.6774 |
| middle | 30 | 0.0000 [0.000, 0.114] | 0.7115 |
| start | 42 | 0.0000 [0.000, 0.084] | 0.8090 |

![Redaction Sensitivity Grid](/home/kinshuk-gupta/KINSHUK GUPTA/IIT BHILAI/IIT BHILAI 7TH SEMESTER/DSL504 - NATURAL LANGUAGE PROCESSING/Group Project/Sudarshana/reports/figures/redaction_sensitivity.png)
*Redaction Sensitivity Grid*

---

## 10. Limitations

1. **Real split sizes differ from team baseline** — P5 evaluates on committed sample splits (test=206, hard_neg=15); team baseline ran on full dataset (test=42 999). Numbers are **not comparable**.
2. **Hard-neg N=15** — only 15 hard-negative records; FPR estimates are indicative only (wide Wilson CIs).
3. **ASR proxy** — PSR measures payload-text survival; real ASR requires LLM execution and behavioural verification.
4. **ROUGE-1 utility proxy** — surface-text overlap; actual task utility requires downstream agent evaluation.
5. **token_head: N/A** — no trained weights; all token-head metrics are N/A.
6. **No obfuscated set** — obfuscated held-out evaluation skipped (P1/P2 deliverable).
7. **No pair-input evaluation** — P3's (user_task, document) pair collator not integrated; all predictors use document-only input.
8. **No LLM ASR** — `LLMBackend` protocol stubbed; `compute_llm_asr` returns N/A.
9. **Sensitivity is SIMULATED** — gold span perturbations, not real predicted spans from a trained model.

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
