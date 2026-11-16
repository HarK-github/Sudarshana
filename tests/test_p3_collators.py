"""Unit tests for SingleStringCollator and PairInputCollator (Task 1.2)."""

import pytest
import torch
from pathlib import Path
from transformers import AutoTokenizer

from src.data.dataset import generate_mock_dataset
from src.data.collators import (
    SingleStringCollator,
    PairInputCollator,
    LABEL2ID,
    PAD_LABEL_ID,
    get_tokenizer,
)


@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer("checkpoints/prototype-token-head-deberta-v3-xsmall-seed42")



def test_single_string_collator(tokenizer):
    """Verify SingleStringCollator produces valid batch tensors and BIO masks."""
    mock_records = [r.to_dict() for r in generate_mock_dataset(n_samples=4)]
    collator = SingleStringCollator(tokenizer=tokenizer, max_length=128)
    batch = collator(mock_records)

    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch
    assert "token_labels" in batch
    assert "attack_type_ids" in batch
    assert "metadata" in batch

    assert batch["input_ids"].shape[0] == 4
    assert batch["labels"].shape == (4,)
    assert batch["token_labels"].shape == batch["input_ids"].shape

    # For any injected sample, verify presence of B-INJ (1)
    has_b_inj = False
    for i in range(4):
        if batch["labels"][i] == 1:
            tok_labels = batch["token_labels"][i].tolist()
            if LABEL2ID["B-INJ"] in tok_labels:
                has_b_inj = True
    assert has_b_inj, "At least one injected record should have B-INJ token"


def test_pair_input_collator_segment_masking(tokenizer):
    """Verify PairInputCollator masks task tokens with -100 and aligns chunk spans."""
    mock_records = [r.to_dict() for r in generate_mock_dataset(n_samples=6)]
    collator = PairInputCollator(
        tokenizer=tokenizer,
        max_task_length=32,
        max_chunk_length=128,
        total_max_length=160,
    )
    batch = collator(mock_records)

    input_ids = batch["input_ids"]
    token_labels = batch["token_labels"]
    labels = batch["labels"]

    assert input_ids.shape[0] == 6
    assert input_ids.shape[1] <= 160
    assert token_labels.shape == input_ids.shape

    for i in range(6):
        # Decode and check sequence IDs
        enc = tokenizer(
            mock_records[i]["user_task"],
            mock_records[i]["text"],
            truncation="only_second",
            max_length=160,
            padding=False,
        )
        seq_ids = enc.sequence_ids()
        # All non-chunk tokens (seq_id != 1) must be PAD_LABEL_ID (-100)
        for t_idx, s_id in enumerate(seq_ids):
            if s_id != 1:
                assert token_labels[i][t_idx] == PAD_LABEL_ID, (
                    f"Token at index {t_idx} with seq_id={s_id} must have label -100, got {token_labels[i][t_idx]}"
                )

        if labels[i] == 1:
            # Positive sample must have B-INJ
            tok_labels_list = token_labels[i].tolist()
            assert (LABEL2ID["B-INJ"] in tok_labels_list), "Injected sample must have B-INJ in pair input"
        elif mock_records[i]["attack_category"] == "hard_negative":
            # Hard negative should have NO B-INJ or I-INJ
            tok_labels_list = [x for x in token_labels[i].tolist() if x != PAD_LABEL_ID]
            assert all(x == LABEL2ID["O"] for x in tok_labels_list), (
                "Hard negative should contain only O tokens in target segment"
            )
