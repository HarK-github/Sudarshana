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

- **Recall by Placement**: Start: **100.0%** (42/42) | Middle: **96.67%** (29/30) | End: **100.0%** (53/53). Sliding windows completely eliminate truncation issues past token 512.
- **Short Documents (`< 500` chars, $n = 14$)**: Selective redaction rate: **28.6%**, Block rate: **71.4%**, Retained content: **21.6%**.
- **Medium Documents (`500–1500` chars, $n = 76$)**: Selective redaction rate: **65.8%**, Block rate: **34.2%**, Retained content: **54.9%**.
- **Long Documents (`> 1500` chars, $n = 35$)**: Selective redaction rate: **88.6%**, Block rate: **11.4%**, Retained content: **78.8%**.
- **Clean Counterfactuals ($n = 125$)**: Clean documents blocked: **11.20%** [6.79%, 17.92%]; clean content preserved: **76.19%** [70.67%, 81.26%].
- **Inference Latency (CPU, 16 vCPUs)**: Median **0.871s** per 1,362-token document (p95: **1.040s**; normalized 1.5k tokens: **0.959s**).

---

## 4. Error Analysis

Out of 125 test attacks, 118 (94.4%) were completely sanitized or blocked with zero residual characters. The 7 surviving cases break down as:

1. **Whitespace Delimiters (3 samples — Indices 31, 124, 199)**: Exactly 1 character (`\n`) where the model cleanly stopped at the word boundary without deleting the surrounding newline. Harmless.
2. **Subtle Phrasing (1 sample — Index 3)**: 13 characters (`\nProvide insi`) survived in a table cell where token confidence hovered at 0.34 at the beginning of the sentence.
3. **Multi-line Code Payloads (2 samples — Indices 28, 87)**: 23 characters (scapy payload) and 224 characters (Python multiprocessing loop inside a docstring). The directive header was excised, but trailing loop lines exceeded the $2.5\times$ relative expansion cap.
4. **Multi-Sentence Prompt (1 sample — Index 66)**: 42 characters surviving from a secondary sentence in a multi-sentence feedback attack.

---

## 5. Integration Contracts

- **Data Loader**: Consumes schema records `(user_task, document, label, injection_span, attack_type)` supporting both dict spans `[{"start": s, "end": e}]` and list spans `[[s, e]]`.
- **Text Normalization**: `predict_spans` operates on raw character offsets. When the normalizer is active, wrap predicted spans with `offset_map.map_span(s, e)` before passing to `redact()`.
- **Joint Model Training**: Import `TokenClassificationHead` from `sudarshana.models.token_head`. Multi-task joint loss combines detector loss and token loss via `SudarshanaTokenClassifier.compute_joint_loss(loss_detect, loss_tok, lambda_tok=0.5)`.
- **Downstream Generation**: Pass sanitized documents from `redact(doc, spans)` directly into LLM prompts for downstream RAG task completion.
