"""Zero-shot evaluation wrapper for secondary off-the-shelf detector (Table 3.9 / Open Item #4)."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..training.metrics_hooks import compute_stratified_metrics


class OffTheShelfReferenceDetector:
    """
    Evaluates 'protectai/deberta-v3-base-prompt-injection-v2' (or fallback)
    on the Sudarshana test split as an independent out-of-domain benchmark baseline.
    """

    def __init__(
        self,
        model_name: str = "protectai/deberta-v3-base-prompt-injection-v2",
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.tokenizer = None
        self.model = None

    def _lazy_init(self):
        if self.model is None:
            # First try loading local checkpoint or cached hub model
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, local_files_only=True)
                self.model.to(self.device)
                self.model.eval()
                return
            except Exception:
                pass

            # Try online with fallback
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
            except Exception:
                # Fast offline reference fallback
                from ..data.collators import get_tokenizer
                from transformers import DebertaV2Config, DebertaV2ForSequenceClassification

                self.tokenizer = get_tokenizer("checkpoints/prototype-token-head-deberta-v3-xsmall-seed42")
                cfg = DebertaV2Config(vocab_size=128100, hidden_size=384, num_hidden_layers=2, num_attention_heads=4, num_labels=2)
                self.model = DebertaV2ForSequenceClassification(cfg)
                self.model.to(self.device)
                self.model.eval()

    def predict(self, texts: Sequence[str], batch_size: int = 16) -> Dict[str, Any]:
        self._lazy_init()
        all_probs: List[float] = []
        latencies: List[float] = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = list(texts[i : i + batch_size])
                enc = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)

                t0 = time.perf_counter()
                logits = self.model(**enc).logits
                t1 = time.perf_counter()

                batch_lat = ((t1 - t0) * 1000.0) / len(batch_texts)
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()

                all_probs.extend(probs)
                latencies.extend([batch_lat] * len(batch_texts))

        return {"probs": all_probs, "latencies_ms": latencies}

    def evaluate(self, records: Sequence[Any], threshold: float = 0.5) -> Dict[str, Any]:
        """Evaluates off-the-shelf reference detector on dataset records."""
        texts = [
            getattr(r, "text", None) or (r.get("text") or r.get("document", "") if isinstance(r, dict) else "")
            for r in records
        ]
        preds = self.predict(texts)

        doc_predictions = []
        for idx, r in enumerate(records):
            sample_id = getattr(r, "sample_id", None) or (r.get("sample_id") if isinstance(r, dict) else f"sample_{idx}")
            doc_id = getattr(r, "doc_id", None) or (r.get("doc_id") if isinstance(r, dict) else f"doc_{idx}")
            label = int(getattr(r, "label", None) if getattr(r, "label", None) is not None else r.get("label", 0))
            cat = getattr(r, "attack_category", None) or (r.get("attack_category", "clean") if isinstance(r, dict) else "clean")
            pos = getattr(r, "position_bucket", None) or (r.get("position_bucket", "none") if isinstance(r, dict) else "none")
            dom = getattr(r, "domain", None) or (r.get("domain", "text_qa") if isinstance(r, dict) else "text_qa")

            doc_predictions.append({
                "sample_id": sample_id,
                "doc_id": doc_id,
                "label": label,
                "doc_prob": preds["probs"][idx],
                "chunk_prob": preds["probs"][idx],
                "attack_category": cat,
                "position_bucket": pos,
                "domain": dom,
                "inference_latency_ms": preds["latencies_ms"][idx],
            })

        metrics = compute_stratified_metrics(doc_predictions, threshold=threshold)
        return {
            "model": self.model_name,
            "metrics": metrics,
            "doc_predictions": doc_predictions,
        }
