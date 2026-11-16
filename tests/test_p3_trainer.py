"""Unit tests for UnifiedTrainer loop, differential optimizer, and artifact export (Task 1.5)."""

import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from src.data.dataset import PromptInjectionDataset, generate_mock_dataset
from src.data.collators import SingleStringCollator, get_tokenizer
from src.models.deberta_multitask import DebertaMultiTaskFilter, DebertaMultiTaskConfig
from src.training.trainer import UnifiedTrainer, TrainingArguments, create_optimizer


class DummyEncoder(nn.Module):
    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_size})()
        self.embedding = nn.Embedding(128005, hidden_size)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        x = self.embedding(input_ids)
        return type("Output", (), {"last_hidden_state": x})()


def test_create_optimizer_groups():
    """Verify differential learning rates and no-decay lists."""
    hidden_size = 64
    config = DebertaMultiTaskConfig(hidden_size=hidden_size)
    dummy = DummyEncoder(hidden_size=hidden_size)
    model = DebertaMultiTaskFilter(config=config, encoder=dummy)

    opt = create_optimizer(model, backbone_lr=2e-5, heads_lr=1e-4, weight_decay=0.01)
    assert len(opt.param_groups) >= 2
    lrs = [g["lr"] for g in opt.param_groups]
    assert 2e-5 in lrs
    assert 1e-4 in lrs


def test_unified_trainer_step_and_export():
    """Verify complete train -> eval -> export cycle."""
    mock_train = generate_mock_dataset(n_samples=8)
    mock_val = generate_mock_dataset(n_samples=4)

    train_ds = PromptInjectionDataset(mock_train)
    val_ds = PromptInjectionDataset(mock_val)

    tokenizer = get_tokenizer("checkpoints/prototype-token-head-deberta-v3-xsmall-seed42")
    collator = SingleStringCollator(tokenizer=tokenizer, max_length=64)

    hidden_size = 64
    config = DebertaMultiTaskConfig(hidden_size=hidden_size)
    dummy = DummyEncoder(hidden_size=hidden_size)
    model = DebertaMultiTaskFilter(config=config, encoder=dummy)

    with tempfile.TemporaryDirectory() as tmp_dir:
        args = TrainingArguments(
            output_dir=tmp_dir,
            seed=42,
            epochs=1,
            per_device_batch_size=4,
            gradient_accumulation_steps=1,
            backbone_lr=1e-4,
            heads_lr=1e-3,
            step_name="step1_test",
            device="cpu",
            fp16=False,
        )

        trainer = UnifiedTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            val_dataset=val_ds,
            collate_fn=collator,
        )

        train_res = trainer.train()
        assert "history" in train_res
        assert len(train_res["history"]) == 1

        eval_res = trainer.evaluate(val_ds, split_name="val")
        assert "metrics" in eval_res
        assert "overall" in eval_res["metrics"]
        assert "recall" in eval_res["metrics"]["overall"]

        # Check export files
        expected_json = Path(tmp_dir) / "predictions_step1_test_val_seed42.json"
        assert expected_json.exists()
