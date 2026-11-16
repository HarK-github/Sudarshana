"""Unified mixed-precision training and validation loop (Section 5)."""

from __future__ import annotations

import json
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

from .metrics_hooks import (
    aggregate_document_predictions,
    compute_classification_metrics,
    compute_stratified_metrics,
)


@dataclass
class TrainingArguments:
    output_dir: str = "checkpoints/ablation_run"
    seed: int = 42
    epochs: int = 3
    per_device_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    backbone_lr: float = 3e-5
    heads_lr: float = 1.5e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    fp16: bool = True
    logging_steps: int = 10
    eval_steps: int = 50
    step_name: str = "step1"
    pooling_method: str = "max"
    top_k: int = 2
    use_token_head: bool = False
    use_type_head: bool = False
    device: Optional[str] = None


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_optimizer(
    model: nn.Module,
    backbone_lr: float = 2e-5,
    heads_lr: float = 1e-4,
    weight_decay: float = 0.01,
) -> torch.optim.AdamW:
    """
    Creates AdamW optimizer with differential learning rates and weight decay exclusions.
    - Excludes weight decay on bias and LayerNorm parameters.
    - Uses higher learning rate for task-specific classification heads.
    """
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]

    backbone_params_decay = []
    backbone_params_no_decay = []
    head_params_decay = []
    head_params_no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_head = any(h in name for h in ["detect_head", "token_head", "type_head"])
        is_no_decay = any(nd in name for nd in no_decay)

        if is_head:
            if is_no_decay:
                head_params_no_decay.append(param)
            else:
                head_params_decay.append(param)
        else:
            if is_no_decay:
                backbone_params_no_decay.append(param)
            else:
                backbone_params_decay.append(param)

    optimizer_grouped_parameters = [
        {"params": backbone_params_decay, "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": backbone_params_no_decay, "lr": backbone_lr, "weight_decay": 0.0},
        {"params": head_params_decay, "lr": heads_lr, "weight_decay": weight_decay},
        {"params": head_params_no_decay, "lr": heads_lr, "weight_decay": 0.0},
    ]

    # Filter out empty parameter groups
    optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if len(g["params"]) > 0]

    return torch.optim.AdamW(optimizer_grouped_parameters, betas=(0.9, 0.999), eps=1e-8)


class UnifiedTrainer:
    """
    Unified Trainer executing reproducible training, validation, pooling,
    and prediction artifact exports matching Section 3.3.
    """

    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        collate_fn: Optional[Callable] = None,
    ):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.collate_fn = collate_fn

        set_seed(self.args.seed)

        if self.args.device:
            self.device = torch.device(self.args.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        # Mixed precision setup
        self.use_amp = self.args.fp16 and (self.device.type == "cuda")
        if self.use_amp:
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None

        self.output_dir = Path(self.args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> Dict[str, Any]:
        """Runs the standardized training loop."""
        if self.train_dataset is None:
            raise ValueError("train_dataset must be provided for training.")

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            drop_last=False,
        )

        total_steps = len(train_loader) * self.args.epochs // self.args.gradient_accumulation_steps
        warmup_steps = max(1, int(total_steps * self.args.warmup_ratio))

        optimizer = create_optimizer(
            self.model,
            backbone_lr=self.args.backbone_lr,
            heads_lr=self.args.heads_lr,
            weight_decay=self.args.weight_decay,
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max(1, total_steps),
        )

        history: List[Dict[str, Any]] = []
        global_step = 0

        self.model.train()
        for epoch in range(1, self.args.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                token_labels = batch["token_labels"].to(self.device) if "token_labels" in batch else None
                attack_type_ids = batch["attack_type_ids"].to(self.device) if "attack_type_ids" in batch else None

                autocast_ctx = torch.amp.autocast("cuda") if self.use_amp else nullcontext()
                with autocast_ctx:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        token_labels=token_labels,
                        attack_type_ids=attack_type_ids,
                    )
                    loss = outputs["loss"] / self.args.gradient_accumulation_steps

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    epoch_loss += loss.item() * self.args.gradient_accumulation_steps

                    if (step + 1) % self.args.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                        self.scaler.step(optimizer)
                        self.scaler.update()
                        scheduler.step()
                        optimizer.zero_grad()
                        global_step += 1
                else:
                    loss.backward()
                    epoch_loss += loss.item() * self.args.gradient_accumulation_steps

                    if (step + 1) % self.args.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        global_step += 1

            avg_epoch_loss = epoch_loss / max(1, len(train_loader))
            log_entry = {"epoch": epoch, "train_loss": avg_epoch_loss, "global_step": global_step}

            # Run validation if available
            if self.val_dataset is not None:
                val_res = self.evaluate(self.val_dataset, split_name="val")
                log_entry["val_metrics"] = val_res["metrics"]["overall"]
                v_probs = [float(p["doc_prob"]) for p in val_res["doc_predictions"]]
                if v_probs:
                    v_min = float(min(v_probs))
                    v_mean = float(sum(v_probs) / len(v_probs))
                    v_max = float(max(v_probs))
                    print(f"  [Epoch {epoch} Val] Train Loss: {avg_epoch_loss:.4f} | Val Probs: min={v_min:.4f}, mean={v_mean:.4f}, max={v_max:.4f}")

            history.append(log_entry)

        # Save trained weights
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(self.output_dir)

        # Save training history
        with open(self.output_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        return {"history": history, "final_loss": avg_epoch_loss}

    def evaluate(
        self,
        dataset: Dataset,
        split_name: str = "test",
        export_artifacts: bool = True,
    ) -> Dict[str, Any]:
        """
        Runs evaluation, collects predictions, pools over doc_id, and exports Parquet artifact.
        """
        data_loader = DataLoader(
            dataset,
            batch_size=self.args.per_device_batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

        self.model.eval()
        raw_chunk_preds: List[Dict[str, Any]] = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                t0 = time.perf_counter()
                autocast_ctx = torch.amp.autocast("cuda") if self.use_amp else nullcontext()
                with autocast_ctx:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                t1 = time.perf_counter()
                latency_per_sample_ms = ((t1 - t0) * 1000.0) / len(input_ids)

                chunk_probs = outputs["chunk_probs"].cpu().numpy()
                token_probs = outputs["token_probs"].cpu().numpy() if outputs["token_probs"] is not None else None
                metadata = batch["metadata"]
                batch_labels = batch["labels"].cpu().numpy()

                for b_idx in range(len(input_ids)):
                    meta = metadata[b_idx]
                    t_probs_item = token_probs[b_idx].tolist() if token_probs is not None else []
                    t_preds_item = (
                        (outputs["token_logits"][b_idx].argmax(dim=-1).cpu().tolist())
                        if outputs["token_logits"] is not None
                        else []
                    )

                    raw_chunk_preds.append({
                        "sample_id": meta["sample_id"],
                        "doc_id": meta["doc_id"],
                        "label": int(batch_labels[b_idx]),
                        "chunk_prob": float(chunk_probs[b_idx]),
                        "token_probs": t_probs_item,
                        "token_preds": t_preds_item,
                        "attack_category": meta.get("attack_category", "clean"),
                        "position_bucket": meta.get("position_bucket", "none"),
                        "domain": meta.get("domain", "text_qa"),
                        "inference_latency_ms": latency_per_sample_ms,
                    })

        # Document-level aggregation
        doc_predictions = aggregate_document_predictions(
            raw_chunk_preds,
            pooling_method=self.args.pooling_method,
            top_k=self.args.top_k,
        )

        metrics = compute_stratified_metrics(doc_predictions, threshold=0.5, score_key="doc_prob")

        # Export standardized predictions artifact (Section 3.3)
        if export_artifacts:
            self._export_predictions(doc_predictions, chunk_predictions=raw_chunk_preds, split_name=split_name)

        return {
            "metrics": metrics,
            "doc_predictions": doc_predictions,
            "chunk_predictions": raw_chunk_preds,
        }

    def _export_predictions(
        self,
        doc_predictions: List[Dict[str, Any]],
        chunk_predictions: Optional[List[Dict[str, Any]]] = None,
        split_name: str = "test",
    ):
        """Exports predictions_{step}_{split}_seed{s}.parquet and token predictions adhering to Section 3.3."""
        filename_base = f"predictions_{self.args.step_name}_{split_name}_seed{self.args.seed}"
        out_parquet = self.output_dir / f"{filename_base}.parquet"
        out_json = self.output_dir / f"{filename_base}.json"

        # Prepare export rows
        export_rows = []
        for d in doc_predictions:
            export_rows.append({
                "sample_id": str(d["sample_id"]),
                "doc_id": str(d["doc_id"]),
                "label": int(d["label"]),
                "attack_category": str(d["attack_category"]),
                "position_bucket": str(d["position_bucket"]),
                "domain": str(d["domain"]),
                "chunk_prob": float(max(d["chunk_probs"])) if d["chunk_probs"] else 0.0,
                "doc_prob": float(d["doc_prob"]),
                "num_chunks": int(d["num_chunks"]),
                "inference_latency_ms": float(d["inference_latency_ms"]),
            })

        # Always save JSON for high portability
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(export_rows, f, indent=2)

        # Export token predictions if token head is enabled
        if chunk_predictions is not None and self.args.use_token_head:
            tok_filename = f"token_predictions_{self.args.step_name}_{split_name}_seed{self.args.seed}.json"
            tok_out = self.output_dir / tok_filename
            tok_rows = []
            for cp in chunk_predictions:
                tok_rows.append({
                    "sample_id": cp.get("sample_id"),
                    "doc_id": cp.get("doc_id"),
                    "label": cp.get("label"),
                    "chunk_prob": cp.get("chunk_prob"),
                    "token_probs": cp.get("token_probs", []),
                    "token_preds": cp.get("token_preds", []),
                    "attack_category": cp.get("attack_category", "clean"),
                })
            with open(tok_out, "w", encoding="utf-8") as f:
                json.dump(tok_rows, f, indent=2)

        # Save Parquet via isolated process to avoid Windows pyarrow thread hook conflicts
        try:
            import subprocess
            import sys
            cmd = [
                sys.executable,
                "-c",
                f'import pandas as pd; pd.read_json(r"{out_json}").to_parquet(r"{out_parquet}", index=False)',
            ]
            subprocess.run(cmd, capture_output=True, check=True)
        except Exception:
            pass
