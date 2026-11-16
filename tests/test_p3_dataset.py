"""Unit tests for P3 dataset module (Task 1.1)."""

import pytest
from pathlib import Path
from src.data.dataset import (
    PromptInjectionDataset,
    PromptInjectionRecord,
    generate_mock_dataset,
    raw_dict_to_record,
)


def test_mock_dataset_generation():
    """Verify mock dataset matches Section 3.1 contract."""
    records = generate_mock_dataset(n_samples=16)
    assert len(records) == 16

    clean_count = sum(1 for r in records if r.label == 0 and r.attack_category == "clean")
    hard_neg_count = sum(1 for r in records if r.label == 0 and r.attack_category == "hard_negative")
    injected_count = sum(1 for r in records if r.label == 1)

    assert clean_count > 0
    assert hard_neg_count > 0
    assert injected_count > 0

    for r in records:
        assert isinstance(r.sample_id, str)
        assert isinstance(r.doc_id, str)
        assert isinstance(r.user_task, str)
        assert isinstance(r.text, str)
        assert r.label in (0, 1)
        assert r.attack_category in {"clean", "hard_negative", "task_irrelevant", "task_relevant", "targeted"}
        assert r.position_bucket in {"start", "middle", "end", "none"}
        assert r.domain in {"text_qa", "code_qa"}

        if r.label == 1:
            assert len(r.injection_char_spans) > 0
            for start, end in r.injection_char_spans:
                span_text = r.text[start:end]
                assert "SYSTEM OVERRIDE" in span_text or len(span_text) > 0
        else:
            assert len(r.injection_char_spans) == 0


def test_dataset_wrapper_with_mock():
    """Verify PyTorch Dataset wrapper indexing and dictionary contract."""
    mock_records = generate_mock_dataset(n_samples=8)
    ds = PromptInjectionDataset(mock_records)
    assert len(ds) == 8

    item = ds[0]
    required_keys = {
        "sample_id", "doc_id", "user_task", "text", "label",
        "injection_char_spans", "attack_type_id", "attack_category",
        "position_bucket", "domain"
    }
    assert required_keys.issubset(item.keys())


def test_real_split_ingestion():
    """Verify ingestion of P1 train.jsonl data file."""
    train_path = Path("data/splits/train.jsonl")
    if not train_path.exists():
        pytest.skip("train.jsonl not found in data/splits")

    ds = PromptInjectionDataset(train_path)
    assert len(ds) > 0
    sample = ds[0]
    assert "user_task" in sample
    assert "text" in sample
    assert sample["label"] in (0, 1)

