# Model Card — Sudarshana Joint Detector (P3)

## Model Summary

| Field | Value |
|---|---|
| **Model ID** | `sudarshana-joint-deberta-v3-xsmall` |
| **Backbone** | `microsoft/deberta-v3-xsmall` |
| **Task** | Indirect prompt-injection detection + span localisation |
| **Input format** | `[CLS] user_task [SEP] chunk [SEP]` (pair) or `[CLS] chunk [SEP]` |
| **Heads** | Chunk-level binary classifier + token-level BIO tagger |
| **Status** | Frozen final checkpoint |

---

## Architecture

```
[CLS] user_task [SEP] chunk [SEP]
         │
   DeBERTa-v3-xsmall encoder (shared)
         │
   ┌─────┴──────────────────────┐
[CLS] hidden              All token hiddens
   │                             │
Detection head           Token head
(safe / malicious)       (O / B-INJ / I-INJ)
```

**Joint loss:**  `L = L_detect + λ_tok × L_tok`  (default λ_tok = 1.0)

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Backbone | `microsoft/deberta-v3-xsmall` |
| Input format | Pair `[CLS] task [SEP] chunk [SEP]` |
| Chunk size | 256 tokens |
| Stride | 64 tokens |
| Task prefix budget | 48 tokens |
| Epochs | 3 |
| Learning rate | 2e-5 (linear warmup 10%) |
| Batch size | 8 |
| λ_tok | 1.0 |
| Seeds | 42, 43, 44 |
| Augmentation | Obfuscation (base64, homoglyphs, zero-width, leet, rand-caps) p=0.5 |
| Detection class weight | 2.0 for positive |
| Token class weights | O=1.0, B-INJ=8.0, I-INJ=8.0 |

---

## Data

| Split | Source |
|---|---|
| Train | `data/splits/train.jsonl` (BIPIA + AgentDojo + TensorTrust) |
| Val | `data/splits/val.jsonl` |
| Test | `data/splits/test.jsonl` |
| Hard negatives | `data/splits/hard_negatives.jsonl` |

Training injection records are augmented with obfuscations on-the-fly. Clean records (label=0) are passed through unchanged.

---

## Ablation Conditions

| Condition | Pair input | Augment | Expected Δ |
|---|---|---|---|
| `chunk_only` | ✗ | ✗ | Baseline |
| `pair` | ✓ | ✗ | ↓ FP on task-relevant content |
| `pair_aug` (**final**) | ✓ | ✓ | ↑ Recall on obfuscated attacks |

---

## Evaluation Metrics

- **Overall recall / FPR** (val and test splits)
- **Recall by attack category** (`agent_injection`, `code`, `prompt_extraction`, `targeted`, `task_irrelevant`, `task_relevant`)
- **Recall by source** (agentdojo, bipia, tensortrust)
- **Recall by position** (start, middle, end, tool_output)
- **Token-level F1** (B-INJ, I-INJ, O)
- **Span overlap F1** (IoU > 0 threshold)

---

## Limitations

- Max effective document length is sliding-window bounded; very long documents (>10k tokens) increase latency linearly.
- Obfuscation augmentation covers common patterns but not every novel encoding.
- Pair-input benefit depends on quality of `user_task`; empty tasks fall back to chunk-only automatically.
- Not tested on code-switching or multilingual documents.

---

## Integration

```python
from sudarshana.models.joint_model import SudarshanaJointModel
from transformers import AutoTokenizer
import torch

ckpt = "checkpoints/joint-pair-deberta-v3-xsmall"
tokenizer = AutoTokenizer.from_pretrained(ckpt)
model = SudarshanaJointModel()
model.load_state_dict(torch.load(f"{ckpt}/pytorch_model.bin", map_location="cpu"))
model.eval()

# Pair input
enc = tokenizer(
    user_task, chunk_text,
    max_length=256, truncation="only_second",
    return_tensors="pt"
)
with torch.no_grad():
    out = model(enc["input_ids"], enc["attention_mask"])

chunk_pred = out["det_logits"].argmax(-1).item()   # 0=safe, 1=malicious
tok_preds  = out["tok_logits"].argmax(-1)[0]       # (seq_len,) BIO labels
```

See [`sudarshana/training/joint_train.py`](../sudarshana/training/joint_train.py) for the full training pipeline and [`sudarshana/training/ablation_runner.py`](../sudarshana/training/ablation_runner.py) for ablation experiments.

---

## Checkpoint Contents

```
checkpoints/joint-pair-deberta-v3-xsmall/
├── pytorch_model.bin       ← frozen model weights
├── tokenizer_config.json
├── tokenizer.json
├── special_tokens_map.json
└── run_metadata.json       ← training config + history + recall report
```

---

## Responsible Use

This model is a **detection aid** for RAG pipelines. It is not a security boundary by itself. False negatives are possible — especially for novel injection phrasings not covered in training data. Always combine with other defenses (prompt hardening, output monitoring).

**Owner:** P3 — Model Development  
**Last updated:** 2026-11-15
