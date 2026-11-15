"""Joint DeBERTa-v3 model for Sudarshana — P3 deliverable.

Architecture:
  - Shared DeBERTa-v3-xsmall encoder.
  - Pair-input format: [CLS] user_task [SEP] chunk [SEP]
  - Chunk-level detection head  (binary: safe / malicious).
  - Token-level injection head  (O / B-INJ / I-INJ).
  - Joint loss: L = L_detect + lambda_tok * L_tok.

Usage:
    model = SudarshanaJointModel()
    out   = model(input_ids, attention_mask,
                  chunk_labels=..., token_labels=...)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    from transformers import AutoConfig, AutoModel
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None     # type: ignore[assignment]
    AutoConfig = None   # type: ignore[assignment]
    AutoModel = None    # type: ignore[assignment]
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Sub-heads
# ---------------------------------------------------------------------------

class _DetectionHead(nn.Module):
    """[CLS]-pooled binary classifier (safe=0 / malicious=1)."""

    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_size, 2)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, cls_hidden: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(cls_hidden))          # (B, 2)


class _TokenHead(nn.Module):
    """Per-token classifier (O=0 / B-INJ=1 / I-INJ=2)."""

    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_size, 3)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(hidden))              # (B, L, 3)


# ---------------------------------------------------------------------------
# Joint model
# ---------------------------------------------------------------------------

class SudarshanaJointModel(nn.Module):
    """
    Joint chunk-level + token-level detector.

    Args:
        backbone:    HuggingFace model name / local path.
        lambda_tok:  Weight of token loss in joint objective.
        det_weight:  Class weight for positive (malicious) detection class.
        tok_weights: Class weights for (O, B-INJ, I-INJ) token classes.
    """

    # Label constants
    LABEL_O     = 0
    LABEL_B_INJ = 1
    LABEL_I_INJ = 2

    def __init__(
        self,
        backbone: str = "microsoft/deberta-v3-xsmall",
        lambda_tok: float = 1.0,
        det_weight: float = 2.0,
        tok_weights: Tuple[float, float, float] = (1.0, 8.0, 8.0),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lambda_tok = lambda_tok

        cfg            = AutoConfig.from_pretrained(backbone)
        self.encoder   = AutoModel.from_pretrained(backbone, config=cfg).float()
        self.det_head  = _DetectionHead(cfg.hidden_size, dropout)
        self.tok_head  = _TokenHead(cfg.hidden_size, dropout)

        # Losses
        det_w = torch.tensor([1.0, det_weight], dtype=torch.float)
        tok_w = torch.tensor(list(tok_weights), dtype=torch.float)
        self.det_loss_fct = nn.CrossEntropyLoss(weight=det_w)
        self.tok_loss_fct = nn.CrossEntropyLoss(weight=tok_w, ignore_index=-100)

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        chunk_labels:   Optional[torch.Tensor] = None,   # (B,)  int64 {0,1}
        token_labels:   Optional[torch.Tensor] = None,   # (B,L) int64 {-100,0,1,2}
    ) -> Dict[str, torch.Tensor]:

        enc = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden = enc.last_hidden_state.float()  # (B, L, H)
        cls    = hidden[:, 0, :]                # (B, H)

        det_logits = self.det_head(cls)         # (B, 2)
        tok_logits = self.tok_head(hidden)      # (B, L, 3)

        loss = None
        if chunk_labels is not None and token_labels is not None:
            l_det  = self.det_loss_fct(det_logits, chunk_labels)
            l_tok  = self.tok_loss_fct(
                tok_logits.view(-1, 3),
                token_labels.view(-1),
            )
            loss = l_det + self.lambda_tok * l_tok
        elif chunk_labels is not None:
            loss = self.det_loss_fct(det_logits, chunk_labels)
        elif token_labels is not None:
            loss = self.tok_loss_fct(
                tok_logits.view(-1, 3),
                token_labels.view(-1),
            )

        return {
            "loss":        loss,
            "det_logits":  det_logits,
            "tok_logits":  tok_logits,
            "hidden":      hidden,
        }

    # ------------------------------------------------------------------
    def freeze(self) -> None:
        """Freeze all parameters (call before saving final checkpoint)."""
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(True)
