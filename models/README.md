# Sudarshana Trained Model Weights (Steps 1–6)

All 6 production checkpoints have been frozen and published as release assets under GitHub Release [`v1.0.0-weights`](https://github.com/sujalsom22/Sudarshana/releases/tag/v1.0.0-weights).

---

## 1. Quick Download

To download all 6 model weights directly into their respective `models/final_step*/` directories:

```bash
python scripts/download_weights.py --all
```

Or download only the final Step 6 pipeline model:

```bash
python scripts/download_weights.py --step step6
```

Or using GitHub CLI:

```bash
gh release download v1.0.0-weights --dir .weights_staging
```

---

## 2. Directory Layout & Checkpoint Mapping

| Step | Architecture Variant | Release Asset Name | Target Directory | Size |
| :---: | :--- | :--- | :--- | :---: |
| **1** | Baseline (Single 512, [CLS]) | `step1_baseline_weights.pt` | `models/final_step1_baseline/model_weights.pt` | ~283 MB |
| **2** | + Normalizer (NFKC, ZW, B64) | `step2_normalizer_weights.pt` | `models/final_step2_normalizer/model_weights.pt` | ~283 MB |
| **3** | + Chunking (256 win, top-2) | `step3_chunking_weights.pt` | `models/final_step3_chunking/model_weights.pt` | ~283 MB |
| **4** | + Pair Input (`user_task`) | `step4_pair_input_weights.pt` | `models/final_step4_pair_input/model_weights.pt` | ~283 MB |
| **5** | + Token Head (Joint Loss) | `step5_token_head_weights.pt` | `models/final_step5_token_head/model_weights.pt` | ~283 MB |
| **6** | + Augmentation (Full Pipeline) | `step6_augmented_weights.pt` | `models/final_step6_augmented/model_weights.pt` | ~283 MB |

---

## 3. Loading Weights in Python

```python
from src.models.deberta_multitask import DebertaMultiTaskFilter
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load final frozen Step 6 model
model = DebertaMultiTaskFilter.from_pretrained(
    "models/final_step6_augmented",
    weights_path="models/final_step6_augmented/model_weights.pt",
    device=device
)
model.eval()
```
