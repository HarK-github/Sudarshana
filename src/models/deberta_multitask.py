"""DeBERTa Multi-Task Detection and Token-Classification Architecture (Section 4)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PreTrainedModel


@dataclass
class DebertaMultiTaskConfig:
    backbone: str = "microsoft/deberta-v3-base"
    hidden_size: int = 768
    num_classes: int = 2
    num_token_labels: int = 3  # 0: O, 1: B-INJ, 2: I-INJ
    num_attack_types: int = 25  # 15 text + 10 code training attack types
    dropout_prob: float = 0.1
    use_token_head: bool = False
    use_type_head: bool = False
    lambda_tok: float = 0.25
    lambda_type: float = 0.10
    class_weights_tok: Optional[Tuple[float, float, float]] = (0.25, 2.50, 1.50)
    class_weights_detect: Optional[Tuple[float, float]] = (2.244, 0.643)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DebertaMultiTaskConfig:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class DebertaMultiTaskFilter(nn.Module):
    """
    Unified DeBERTa-v3 Multi-Task Filter:
    - Shared DeBERTa encoder
    - Chunk Detection Head (CLS -> R^{B x 2})
    - Token Classification Head (H -> R^{B x L x 3})
    - Optional Attack-Type Head (CLS -> R^{B x K})
    """

    def __init__(
        self,
        config: Optional[DebertaMultiTaskConfig] = None,
        encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.config = config or DebertaMultiTaskConfig()

        if encoder is not None:
            self.encoder = encoder
            hidden_size = getattr(self.encoder.config, "hidden_size", self.config.hidden_size)
            self.config.hidden_size = hidden_size
        else:
            self.encoder, hidden_size = self._init_encoder(self.config.backbone)
            self.config.hidden_size = hidden_size

        self.dropout = nn.Dropout(self.config.dropout_prob)

        # 1. Chunk Detection Head with neutral bias init
        self.detect_head = nn.Linear(self.config.hidden_size, self.config.num_classes)
        nn.init.normal_(self.detect_head.weight, std=0.02)
        nn.init.zeros_(self.detect_head.bias)

        # 2. Decoupled 2-Layer MLP Token Head (prevents gradient interference on [CLS])
        if self.config.use_token_head:
            self.token_head = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size // 2),
                nn.GELU(),
                nn.Dropout(self.config.dropout_prob),
                nn.Linear(self.config.hidden_size // 2, self.config.num_token_labels),
            )
        else:
            self.token_head = nn.Linear(self.config.hidden_size, self.config.num_token_labels)

        # 3. Optional Attack-Type Classification Head
        self.type_head = nn.Linear(self.config.hidden_size, self.config.num_attack_types)

        # Buffers for class weights
        detect_w = torch.tensor(
            self.config.class_weights_detect or (2.244, 0.643), dtype=torch.float
        )
        self.register_buffer("detect_weights", detect_w)

        tok_w = torch.tensor(
            self.config.class_weights_tok or (0.25, 2.50, 1.50), dtype=torch.float
        )
        self.register_buffer("tok_weights", tok_w)

        self.detect_loss_fct = nn.CrossEntropyLoss(weight=self.detect_weights)
        self.token_loss_fct = nn.CrossEntropyLoss(weight=self.tok_weights, ignore_index=-100)
        self.type_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    @staticmethod
    def _init_encoder(backbone: str) -> Tuple[nn.Module, int]:
        """Initializes backbone encoder from local checkpoint, hub, or cached DeBERTa-v3."""
        from transformers import AutoConfig, AutoModel, DebertaV2Config, DebertaV2Model

        # Prefer loading real pretrained weights from local cache / hub
        candidate_ids = [backbone, "microsoft/deberta-v3-xsmall", "microsoft/deberta-v3-base"]
        for model_id in candidate_ids:
            try:
                try:
                    cfg = AutoConfig.from_pretrained(model_id, local_files_only=True)
                    encoder = AutoModel.from_pretrained(model_id, config=cfg, local_files_only=True)
                except Exception:
                    cfg = AutoConfig.from_pretrained(model_id)
                    encoder = AutoModel.from_pretrained(model_id, config=cfg)
                hidden_size = getattr(cfg, "hidden_size", 384)
                return encoder, hidden_size
            except Exception:
                continue

        # Fallback to architectural initialization if offline cache missing
        backbone_str = str(backbone).lower()
        hidden_size = 384 if ("xsmall" in backbone_str or "seed42" in backbone_str) else 768
        num_layers = 4 if ("xsmall" in backbone_str or "seed42" in backbone_str) else 12
        num_heads = 6 if ("xsmall" in backbone_str or "seed42" in backbone_str) else 12

        default_cfg = DebertaV2Config(
            vocab_size=128100,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=512,
        )
        encoder = DebertaV2Model(default_cfg)
        return encoder, default_cfg.hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        token_labels: Optional[torch.Tensor] = None,
        attack_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass computing shared representations and multi-task loss.
        """
        kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        encoder_outputs = self.encoder(**kwargs)
        sequence_output = encoder_outputs.last_hidden_state  # (B, L, d)

        # Extract CLS token state: sequence_output[:, 0, :]
        cls_state = sequence_output[:, 0, :]  # (B, d)

        # 1. Chunk detection logits
        detect_logits = self.detect_head(self.dropout(cls_state))  # (B, 2)
        chunk_probs = torch.softmax(detect_logits, dim=-1)[:, 1]  # P(y=1 | chunk)

        token_logits: Optional[torch.Tensor] = None
        token_probs: Optional[torch.Tensor] = None
        if self.config.use_token_head:
            token_logits = self.token_head(self.dropout(sequence_output))  # (B, L, 3)
            # Sum of P(B-INJ) + P(I-INJ) for sentence redaction
            sm_tok = torch.softmax(token_logits, dim=-1)
            token_probs = sm_tok[:, :, 1] + sm_tok[:, :, 2]  # (B, L)

        type_logits: Optional[torch.Tensor] = None
        if self.config.use_type_head:
            type_logits = self.type_head(self.dropout(cls_state))  # (B, K)

        # Compute Losses
        total_loss: Optional[torch.Tensor] = None
        detect_loss: Optional[torch.Tensor] = None
        token_loss: Optional[torch.Tensor] = None
        type_loss: Optional[torch.Tensor] = None

        if labels is not None:
            if self.detect_loss_fct.weight is not None and self.detect_loss_fct.weight.device != detect_logits.device:
                self.detect_loss_fct.weight = self.detect_loss_fct.weight.to(detect_logits.device)
            detect_loss = self.detect_loss_fct(detect_logits, labels)
            total_loss = detect_loss

            # Token loss component
            if self.config.use_token_head and token_labels is not None and token_logits is not None:
                # token_labels: (B, L), token_logits: (B, L, 3)
                token_loss = self.token_loss_fct(
                    token_logits.view(-1, self.config.num_token_labels),
                    token_labels.view(-1),
                )
                total_loss = total_loss + self.config.lambda_tok * token_loss

            # Attack type loss component (only computed on positive samples where attack_type_ids != -100)
            if self.config.use_type_head and attack_type_ids is not None and type_logits is not None:
                # Mask out negative or unlabeled examples
                valid_mask = attack_type_ids != -100
                if valid_mask.any():
                    type_loss = self.type_loss_fct(
                        type_logits[valid_mask],
                        attack_type_ids[valid_mask],
                    )
                    total_loss = total_loss + self.config.lambda_type * type_loss

        loss_dict: Dict[str, float] = {}
        if detect_loss is not None:
            loss_dict["loss_detect"] = float(detect_loss.detach().item())
        if token_loss is not None:
            loss_dict["loss_tok"] = float(token_loss.detach().item())
        if type_loss is not None:
            loss_dict["loss_type"] = float(type_loss.detach().item())
        if total_loss is not None:
            loss_dict["loss_total"] = float(total_loss.detach().item())

        return {
            "loss": total_loss,
            "loss_dict": loss_dict,
            "detect_loss": detect_loss,
            "token_loss": token_loss,
            "type_loss": type_loss,
            "detect_logits": detect_logits,
            "chunk_logits": detect_logits,
            "detect_probs": chunk_probs,
            "chunk_probs": chunk_probs,
            "tok_logits": token_logits,
            "token_logits": token_logits,
            "tok_inj_probs": token_probs,
            "token_probs": token_probs,
            "type_logits": type_logits,
        }

    def save_pretrained(self, save_directory: Union[str, Path]):
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = save_path / "model_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Save weights
        weights_path = save_path / "model_weights.pt"
        torch.save(self.state_dict(), weights_path)

    @classmethod
    def from_pretrained(
        cls,
        save_directory: Union[str, Path],
        device: Optional[torch.device] = None,
    ) -> DebertaMultiTaskFilter:
        save_path = Path(save_directory)
        config_path = save_path / "model_config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        config = DebertaMultiTaskConfig.from_dict(config_dict)

        model = cls(config=config)
        weights_path = save_path / "model_weights.pt"
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location=device or "cpu")
            model.load_state_dict(state_dict)

        if device is not None:
            model.to(device)
        return model
