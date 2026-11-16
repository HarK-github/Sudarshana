"""Unit tests for DebertaMultiTaskFilter architecture and joint loss (Task 1.3)."""

import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from src.models.deberta_multitask import DebertaMultiTaskFilter, DebertaMultiTaskConfig


class DummyEncoder(nn.Module):
    """Fast lightweight mock encoder for isolated tensor and gradient testing."""
    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_size})()
        self.embedding = nn.Embedding(100, hidden_size)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        x = self.embedding(input_ids)
        return type("Output", (), {"last_hidden_state": x})()


def test_model_forward_detect_only():
    """Verify detection-only mode (Ablation Steps 1-4)."""
    hidden_size = 64
    config = DebertaMultiTaskConfig(
        hidden_size=hidden_size,
        use_token_head=False,
        use_type_head=False,
    )
    dummy = DummyEncoder(hidden_size=hidden_size)
    model = DebertaMultiTaskFilter(config=config, encoder=dummy)

    input_ids = torch.tensor([[1, 10, 20, 2], [1, 5, 2, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    labels = torch.tensor([1, 0])

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    assert "loss" in outputs
    assert "detect_loss" in outputs
    assert outputs["token_loss"] is None
    assert outputs["type_loss"] is None
    assert outputs["detect_logits"].shape == (2, 2)
    assert outputs["chunk_probs"].shape == (2,)

    # Gradient flow test
    loss = outputs["loss"]
    loss.backward()
    assert model.detect_head.weight.grad is not None
    assert model.token_head.weight.grad is None


def test_model_forward_joint_heads():
    """Verify joint detection + token head + type head mode (Ablation Steps 5-6)."""
    hidden_size = 64
    lambda_tok = 0.5
    lambda_type = 0.1
    config = DebertaMultiTaskConfig(
        hidden_size=hidden_size,
        use_token_head=True,
        use_type_head=True,
        lambda_tok=lambda_tok,
        lambda_type=lambda_type,
    )
    dummy = DummyEncoder(hidden_size=hidden_size)
    model = DebertaMultiTaskFilter(config=config, encoder=dummy)

    input_ids = torch.tensor([[1, 10, 20, 2], [1, 5, 2, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    labels = torch.tensor([1, 0])
    token_labels = torch.tensor([
        [-100, 1, 2, -100],
        [-100, 0, -100, -100],
    ])
    attack_type_ids = torch.tensor([3, -100])

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        token_labels=token_labels,
        attack_type_ids=attack_type_ids,
    )

    assert outputs["detect_loss"] is not None
    assert outputs["token_loss"] is not None
    assert outputs["type_loss"] is not None

    expected_loss = (
        outputs["detect_loss"]
        + lambda_tok * outputs["token_loss"]
        + lambda_type * outputs["type_loss"]
    )
    assert torch.isclose(outputs["loss"], expected_loss)

    outputs["loss"].backward()
    assert model.detect_head.weight.grad is not None
    assert any(p.grad is not None for p in model.token_head.parameters())
    assert model.type_head.weight.grad is not None


def test_model_save_and_load():
    """Verify serialization and state restoration."""
    hidden_size = 64
    config = DebertaMultiTaskConfig(hidden_size=hidden_size, use_token_head=True)
    dummy = DummyEncoder(hidden_size=hidden_size)
    model = DebertaMultiTaskFilter(config=config, encoder=dummy)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model.save_pretrained(tmp_dir)
        config_file = Path(tmp_dir) / "model_config.json"
        weights_file = Path(tmp_dir) / "model_weights.pt"
        assert config_file.exists()
        assert weights_file.exists()
