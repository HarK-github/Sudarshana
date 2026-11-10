"""Schema-matching data loader for Sudarshana.

Consumes P1's split files if present, with automatic fallback to Sudarshana prototype splits.
Ensures strict schema compliance: (user_task, document, label, injection_span, attack_type).
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

from .labeling import tokenize_and_align_document, PAD_LABEL_ID


class PromptInjectionDataset(Dataset):
    """
    PyTorch Dataset yielding windowed tokens and aligned BIO labels for prompt-injection detection.
    """
    def __init__(
        self,
        records: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 256,
        stride: int = 64,
        include_user_task: bool = False
    ):
        self.tokenizer = tokenizer
        self.windows = []
        self.original_records = records

        for rec_idx, rec in enumerate(records):
            doc = rec["document"]
            spans = [tuple(s) for s in rec.get("injection_span", [])]
            user_task = rec.get("user_task", "") if include_user_task else None

            doc_windows = tokenize_and_align_document(
                document=doc,
                spans=spans,
                tokenizer=tokenizer,
                user_task=user_task,
                max_length=max_length,
                stride=stride
            )
            for w in doc_windows:
                w["record_idx"] = rec_idx
                self.windows.append(w)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.windows[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
            "record_idx": torch.tensor(item["record_idx"], dtype=torch.long),
            "window_idx": torch.tensor(item["window_idx"], dtype=torch.long),
        }


def load_dataset_split(
    split_name: str,
    p1_dir: str = "data/p1_splits",
    fallback_dir: str = "data/splits"
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Load split file with P1 fallback resolution.
    Returns (records, source_path).
    """
    filename = f"{split_name}.jsonl"
    p1_path = os.path.join(p1_dir, filename)
    fallback_path = os.path.join(fallback_dir, filename)

    if os.path.exists(p1_path):
        active_path = p1_path
        source = "P1 authoritative split"
    elif os.path.exists(fallback_path):
        active_path = fallback_path
        source = "Prototype fallback split (schema-matching)"
    else:
        raise FileNotFoundError(f"Neither P1 split ({p1_path}) nor fallback ({fallback_path}) found.")

    records = []
    with open(active_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} records from {active_path} [{source}]")
    return records, active_path
