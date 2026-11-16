"""PyTorch batch collators for Single-String (Steps 1-2) and Pair-Input (Steps 4-6) training."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast

LABEL2ID = {"O": 0, "B-INJ": 1, "I-INJ": 2}
ID2LABEL = {0: "O", 1: "B-INJ", 2: "I-INJ"}
PAD_LABEL_ID = -100


def get_tokenizer(model_name_or_path: str = "checkpoints/prototype-token-head-deberta-v3-xsmall-seed42") -> Any:
    """
    Robust tokenizer loader supporting both local checkpoints and huggingface hub.
    Safely handles DeBERTa-v2/v3 tokenizers across all transformers versions.
    """
    path_obj = Path(model_name_or_path)
    # Check if local directory contains tokenizer.json
    if path_obj.is_dir() and (path_obj / "tokenizer.json").exists():
        tok_file = str(path_obj / "tokenizer.json")
        return PreTrainedTokenizerFast(
            tokenizer_file=tok_file,
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            unk_token="[UNK]",
            mask_token="[MASK]",
        )

    # Check local fallback prototype checkpoints if relative path not found
    candidate_dirs = [
        Path("checkpoints/prototype-token-head-deberta-v3-xsmall-seed42"),
        Path("Sudarshana/checkpoints/prototype-token-head-deberta-v3-xsmall-seed42"),
        Path(__file__).resolve().parent.parent.parent / "checkpoints/prototype-token-head-deberta-v3-xsmall-seed42",
        Path(__file__).resolve().parent.parent / "checkpoints/prototype-token-head-deberta-v3-xsmall-seed42",
    ]
    for c_dir in candidate_dirs:
        if c_dir.exists() and (c_dir / "tokenizer.json").exists():
            return PreTrainedTokenizerFast(
                tokenizer_file=str(c_dir / "tokenizer.json"),
                pad_token="[PAD]",
                cls_token="[CLS]",
                sep_token="[SEP]",
                unk_token="[UNK]",
                mask_token="[MASK]",
            )

    try:
        return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    except Exception:
        return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)



def compute_chunk_labels_and_spans(
    doc_label: int,
    chunk_char_start: int,
    chunk_char_end: int,
    doc_injection_spans: List[Tuple[int, int]],
    min_overlap_chars: int = 3,
) -> Tuple[int, List[Tuple[int, int]]]:
    """
    Computes chunk-level label and relative injection spans.
    A chunk from an injected document is labeled 1 ONLY if the overlap is >= min_overlap_chars.
    Otherwise, chunk_label is 0 and spans are empty.
    """
    if doc_label == 0 or not doc_injection_spans:
        return 0, []

    local_spans: List[Tuple[int, int]] = []
    for inj_start, inj_end in doc_injection_spans:
        overlap_start = max(chunk_char_start, inj_start)
        overlap_end = min(chunk_char_end, inj_end)
        if overlap_end - overlap_start >= min_overlap_chars:
            local_spans.append((overlap_start - chunk_char_start, overlap_end - chunk_char_start))

    chunk_label = 1 if len(local_spans) > 0 else 0
    return chunk_label, local_spans


def align_bio_labels(
    offsets: List[Tuple[int, int]],
    sequence_ids: List[Optional[int]],
    local_injection_spans: List[Tuple[int, int]],
    use_pair_input: bool = True,
) -> List[int]:
    """
    Maps character injection spans to subword token BIO labels (O=0, B-INJ=1, I-INJ=2).
    user_task tokens and special tokens ([CLS], [SEP], [PAD]) are strictly masked with -100.
    """
    target_seq_id = 1 if use_pair_input else 0
    token_labels: List[int] = []

    for idx, (tok_start, tok_end) in enumerate(offsets):
        seq_id = sequence_ids[idx]
        if seq_id != target_seq_id or (tok_start == 0 and tok_end == 0):
            token_labels.append(PAD_LABEL_ID)
            continue

        tag = 0
        for span_start, span_end in local_injection_spans:
            if max(tok_start, span_start) < min(tok_end, span_end):
                if tok_start <= span_start < tok_end or (
                    len(token_labels) > 0 and token_labels[-1] in (0, PAD_LABEL_ID)
                ):
                    tag = 1  # B-INJ
                else:
                    tag = 2  # I-INJ
                break
        token_labels.append(tag)

    return token_labels


def align_tokens_to_bio(
    offset_mapping: List[Tuple[int, int]],
    sequence_ids: List[Optional[int]],
    word_ids: Optional[List[Optional[int]]],
    spans: List[Tuple[int, int]],
    target_sequence_id: int = 1,
    mask_subword_continuations: bool = True,
) -> List[int]:
    """
    Maps character injection spans to subword token BIO labels (O, B-INJ, I-INJ).
    Enforces Section 3.2 rules:
    - sequence_ids != target_sequence_id (special tokens, padding, user_task tokens) get -100.
    - If mask_subword_continuations is True, subword continuations get -100.
    - First token overlapping an injection span gets B-INJ (1).
    - Subsequent tokens inside the same span get I-INJ (2).
    - Clean non-injected tokens get O (0).
    """
    labels: List[int] = []
    prev_word_id = None
    active_span_idx: Optional[int] = None

    for idx, (char_start, char_end) in enumerate(offset_mapping):
        seq_id = sequence_ids[idx]

        # Ignore tokens not belonging to target text segment (CLS, SEP, PAD, or user_task)
        if seq_id != target_sequence_id or (char_start == 0 and char_end == 0):
            labels.append(PAD_LABEL_ID)
            active_span_idx = None
            continue

        # Optional subword continuation masking
        if word_ids is not None:
            word_id = word_ids[idx]
            if mask_subword_continuations and word_id is not None and word_id == prev_word_id:
                labels.append(PAD_LABEL_ID)
                continue
            prev_word_id = word_id

        # Determine if token overlaps with any injection span
        overlapping_span_idx: Optional[int] = None
        for s_idx, (span_start, span_end) in enumerate(spans):
            # Overlap occurs if token range intersects [span_start, span_end)
            if max(char_start, span_start) < min(char_end, span_end):
                overlapping_span_idx = s_idx
                break

        if overlapping_span_idx is not None:
            if active_span_idx != overlapping_span_idx:
                labels.append(LABEL2ID["B-INJ"])
                active_span_idx = overlapping_span_idx
            else:
                labels.append(LABEL2ID["I-INJ"])
        else:
            labels.append(LABEL2ID["O"])
            active_span_idx = None

    return labels


class SingleStringCollator:
    """
    Collator for Ablation Steps 1-2: 512-truncated, single-string input.
    Input format: [CLS] text [SEP]
    """

    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 512,
        mask_subwords: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_subwords = mask_subwords

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [b["text"] for b in batch]
        labels = [int(b["label"]) for b in batch]
        attack_type_ids = [int(b.get("attack_type_id", -100)) for b in batch]

        # Tokenize single string
        encodings = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_offsets_mapping=True,
            return_tensors=None,  # We process offsets in Python before converting to tensor
        )

        all_token_labels: List[List[int]] = []
        batch_size = len(batch)

        for i in range(batch_size):
            spans = batch[i].get("injection_char_spans") or []
            offset_mapping = encodings["offset_mapping"][i]

            # In single-string input, sequence_ids is 0 for document text, None for CLS/SEP/PAD
            seq_ids = encodings.sequence_ids(i)
            try:
                w_ids = encodings.word_ids(i)
            except Exception:
                w_ids = None

            token_bio = align_tokens_to_bio(
                offset_mapping=offset_mapping,
                sequence_ids=seq_ids,
                word_ids=w_ids,
                spans=spans,
                target_sequence_id=0,  # 0 is the text segment for single-sequence input
                mask_subword_continuations=self.mask_subwords,
            )
            all_token_labels.append(token_bio)

        metadata = [
            {
                "sample_id": b.get("sample_id", f"sample_{i}"),
                "doc_id": b.get("doc_id", f"doc_{i}"),
                "attack_category": b.get("attack_category", "clean"),
                "position_bucket": b.get("position_bucket", "none"),
                "domain": b.get("domain", "text_qa"),
            }
            for i, b in enumerate(batch)
        ]

        result = {
            "input_ids": torch.tensor(encodings["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encodings["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "token_labels": torch.tensor(all_token_labels, dtype=torch.long),
            "attack_type_ids": torch.tensor(attack_type_ids, dtype=torch.long),
            "metadata": metadata,
        }

        if "token_type_ids" in encodings:
            result["token_type_ids"] = torch.tensor(encodings["token_type_ids"], dtype=torch.long)

        return result


class PairInputCollator:
    """
    Collator for Ablation Steps 4-6: Pair-input format.
    Input format: [CLS] user_task [SEP] chunk [SEP]
    Allocations:
      user_task max length = 64
      chunk max length = 256
      total max length = 323 tokens ([CLS] + 64 + [SEP] + 256 + [SEP])
    """

    def __init__(
        self,
        tokenizer: Any,
        max_task_length: int = 64,
        max_chunk_length: int = 256,
        total_max_length: int = 323,
        mask_subwords: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_task_length = max_task_length
        self.max_chunk_length = max_chunk_length
        self.total_max_length = total_max_length
        self.mask_subwords = mask_subwords

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        tasks = [b.get("user_task", "") or "" for b in batch]
        chunks = [b.get("text", "") or "" for b in batch]
        labels = [int(b["label"]) for b in batch]
        attack_type_ids = [int(b.get("attack_type_id", -100)) for b in batch]

        # First truncate tasks if needed to ensure max_task_length cap
        # We can pass (tasks, chunks) with truncation='only_second' or pre-truncate tasks
        task_enc = self.tokenizer(
            tasks,
            truncation=True,
            max_length=self.max_task_length,
            add_special_tokens=False,
        )
        task_texts = [
            self.tokenizer.decode(t_ids, skip_special_tokens=True)
            for t_ids in task_enc["input_ids"]
        ]

        # Now tokenize pair: [CLS] task [SEP] chunk [SEP]
        encodings = self.tokenizer(
            task_texts,
            chunks,
            truncation="only_second",
            max_length=self.total_max_length,
            padding=True,
            return_offsets_mapping=True,
            return_tensors=None,
        )

        all_token_labels: List[List[int]] = []
        batch_size = len(batch)

        for i in range(batch_size):
            spans = batch[i].get("injection_char_spans") or []
            offset_mapping = encodings["offset_mapping"][i]
            seq_ids = encodings.sequence_ids(i)

            try:
                w_ids = encodings.word_ids(i)
            except Exception:
                w_ids = None

            # In pair input, sequence_ids == 1 corresponds strictly to the chunk text
            token_bio = align_tokens_to_bio(
                offset_mapping=offset_mapping,
                sequence_ids=seq_ids,
                word_ids=w_ids,
                spans=spans,
                target_sequence_id=1,  # 1 is the chunk segment
                mask_subword_continuations=self.mask_subwords,
            )
            all_token_labels.append(token_bio)

        metadata = [
            {
                "sample_id": b.get("sample_id", f"sample_{i}"),
                "doc_id": b.get("doc_id", f"doc_{i}"),
                "attack_category": b.get("attack_category", "clean"),
                "position_bucket": b.get("position_bucket", "none"),
                "domain": b.get("domain", "text_qa"),
            }
            for i, b in enumerate(batch)
        ]

        result = {
            "input_ids": torch.tensor(encodings["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encodings["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "token_labels": torch.tensor(all_token_labels, dtype=torch.long),
            "attack_type_ids": torch.tensor(attack_type_ids, dtype=torch.long),
            "metadata": metadata,
        }

        if "token_type_ids" in encodings:
            result["token_type_ids"] = torch.tensor(encodings["token_type_ids"], dtype=torch.long)

        return result
