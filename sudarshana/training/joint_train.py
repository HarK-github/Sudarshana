"""P3 Joint Training Pipeline — Sudarshana.

Trains SudarshanaJointModel with:
  - Pair-input format: [CLS] user_task [SEP] chunk [SEP]
  - Chunk-level detection head + token-level injection head
  - Obfuscation augmentation on training set
  - Ablation mode (--ablation chunk_only | pair | no_augment)
  - Saves frozen final checkpoint + run metadata

Usage:
    cd /path/to/Sudarshana
    python -m sudarshana.training.joint_train \\
        --mode pair \\
        --epochs 3 \\
        --backbone microsoft/deberta-v3-xsmall \\
        --output_dir checkpoints/joint-pair-deberta-v3-xsmall
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    from ..models.joint_model import SudarshanaJointModel
    _TORCH_AVAILABLE = True
    _no_grad = torch.no_grad
except (ImportError, AttributeError):  # pragma: no cover
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment,misc]
    AutoTokenizer = None  # type: ignore[assignment]
    get_linear_schedule_with_warmup = None  # type: ignore[assignment]
    SudarshanaJointModel = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False
    import contextlib
    _no_grad = contextlib.nullcontext  # type: ignore[assignment]

# _no_grad_decorator: works as @decorator for both torch and no-torch environments
def _no_grad_wrap(fn):
    """Wraps a function with torch.no_grad if available, else is a no-op."""
    if _TORCH_AVAILABLE:
        return torch.no_grad()(fn)
    return fn

from ..data.augmentation import augment_records
from ..data.labeling import PAD_LABEL_ID, align_spans_to_token_labels
from ..data.loader import load_dataset_split


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PairInjectionDataset(Dataset):
    """
    Sliding-window dataset with pair-input support.

    Pair format: [CLS] user_task [SEP] chunk [SEP]
    Chunk-only : [CLS] chunk [SEP]
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        tokenizer,
        max_length: int = 256,
        stride: int = 64,
        pair_input: bool = True,
        max_task_tokens: int = 48,
    ) -> None:
        self.windows: List[Dict[str, Any]] = []
        self.tokenizer = tokenizer

        for rec_idx, rec in enumerate(records):
            doc    = rec["document"]
            label  = int(rec.get("label", 0))
            task   = rec.get("user_task", "") if pair_input else ""

            raw_spans = rec.get("injection_span") or []
            spans: List[Tuple[int, int]] = []
            for s in raw_spans:
                if isinstance(s, dict):
                    spans.append((int(s["start"]), int(s["end"])))
                elif isinstance(s, (list, tuple)):
                    spans.append((int(s[0]), int(s[1])))

            wins = self._make_windows(
                doc, task, spans, label, rec_idx,
                tokenizer, max_length, stride, max_task_tokens,
            )
            self.windows.extend(wins)

    @staticmethod
    def _make_windows(
        document: str,
        task: str,
        spans: List[Tuple[int, int]],
        doc_label: int,
        rec_idx: int,
        tokenizer,
        max_length: int,
        stride: int,
        max_task_tokens: int,
    ) -> List[Dict[str, Any]]:
        windows = []

        if task.strip():
            # Tokenize task prefix (truncated hard at max_task_tokens)
            task_enc = tokenizer(
                task,
                add_special_tokens=False,
                truncation=True,
                max_length=max_task_tokens,
            )
            task_ids  = task_enc["input_ids"]
            task_mask = task_enc["attention_mask"]
            # Remaining budget for chunk content
            chunk_budget = max_length - len(task_ids) - 3  # [CLS] task [SEP] chunk [SEP]
        else:
            task_ids  = []
            task_mask = []
            chunk_budget = max_length - 2                   # [CLS] chunk [SEP]

        # Tokenize document without specials, with offsets
        doc_enc = tokenizer(
            document,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
        )
        doc_ids    = doc_enc["input_ids"]
        doc_mask   = doc_enc["attention_mask"]
        doc_offs   = doc_enc["offset_mapping"]
        doc_wids   = doc_enc.word_ids()

        # Align token labels over full document
        all_labels = align_spans_to_token_labels(doc_offs, doc_wids, spans)

        step = max(chunk_budget - stride, 1)

        for start in range(0, max(len(doc_ids), 1), step):
            end = min(start + chunk_budget, len(doc_ids))

            chunk_ids   = doc_ids[start:end]
            chunk_mask  = doc_mask[start:end]
            chunk_labs  = all_labels[start:end]

            # Build pair input
            if task_ids:
                # [CLS] + task + [SEP] + chunk + [SEP]
                sep_id = tokenizer.sep_token_id
                cls_id = tokenizer.cls_token_id or tokenizer.bos_token_id or 1

                input_ids = (
                    [cls_id] + task_ids + [sep_id] + chunk_ids + [sep_id]
                )
                attn_mask = (
                    [1] + task_mask + [1] + chunk_mask + [1]
                )
                # Token labels: special/task tokens → -100; chunk tokens kept
                prefix_len = 1 + len(task_ids) + 1  # CLS + task + SEP
                token_labs  = (
                    [PAD_LABEL_ID] * prefix_len
                    + chunk_labs
                    + [PAD_LABEL_ID]           # trailing SEP
                )
            else:
                input_ids  = tokenizer.build_inputs_with_special_tokens(chunk_ids)
                attn_mask  = [1] * len(input_ids)
                token_labs = (
                    [PAD_LABEL_ID]
                    + chunk_labs
                    + [PAD_LABEL_ID]
                )

            # Pad to max_length
            pad_id  = tokenizer.pad_token_id or 0
            pad_len = max_length - len(input_ids)
            if pad_len > 0:
                input_ids  += [pad_id]       * pad_len
                attn_mask  += [0]            * pad_len
                token_labs += [PAD_LABEL_ID] * pad_len

            # Chunk-level label: positive if any injection token in this window
            chunk_lbl = 1 if any(l in (1, 2) for l in chunk_labs) else 0

            windows.append({
                "input_ids":    input_ids[:max_length],
                "attention_mask": attn_mask[:max_length],
                "token_labels": token_labs[:max_length],
                "chunk_label":  chunk_lbl,
                "doc_label":    doc_label,
                "rec_idx":      rec_idx,
                "win_idx":      len(windows),
            })

            if end == len(doc_ids):
                break

        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        w = self.windows[idx]
        return {
            "input_ids":      torch.tensor(w["input_ids"],      dtype=torch.long),
            "attention_mask": torch.tensor(w["attention_mask"], dtype=torch.long),
            "token_labels":   torch.tensor(w["token_labels"],   dtype=torch.long),
            "chunk_label":    torch.tensor(w["chunk_label"],    dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_epoch(model, loader, optimizer, scheduler, device) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        input_ids  = batch["input_ids"].to(device)
        attn_mask  = batch["attention_mask"].to(device)
        tok_labels = batch["token_labels"].to(device)
        chk_labels = batch["chunk_label"].to(device)

        optimizer.zero_grad()
        out  = model(input_ids, attn_mask, chunk_labels=chk_labels, token_labels=tok_labels)
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total += loss.item()
    return total / max(len(loader), 1)


@_no_grad_wrap
def _eval_epoch(model, loader, device) -> float:
    model.eval()
    total = 0.0
    for batch in loader:
        input_ids  = batch["input_ids"].to(device)
        attn_mask  = batch["attention_mask"].to(device)
        tok_labels = batch["token_labels"].to(device)
        chk_labels = batch["chunk_label"].to(device)
        out  = model(input_ids, attn_mask, chunk_labels=chk_labels, token_labels=tok_labels)
        total += out["loss"].item()
    return total / max(len(loader), 1)


# ---------------------------------------------------------------------------
# Recall evaluation (chunk-level)
# ---------------------------------------------------------------------------

@_no_grad_wrap
def evaluate_recall_by_category(
    model, tokenizer, records, device, pair_input, max_length=256, stride=64,
) -> Dict[str, Any]:
    """Chunk-level recall aggregated by attack_type / attack_category / source."""
    model.eval()
    from collections import defaultdict

    buckets: Dict[str, Dict[str, list]] = {
        "attack_type": defaultdict(list),
        "attack_category": defaultdict(list),
        "source": defaultdict(list),
    }

    for rec in records:
        label = int(rec.get("label", 0))
        if label != 1:
            continue   # only measure recall on positives

        ds = PairInjectionDataset(
            [rec], tokenizer, max_length=max_length, stride=stride, pair_input=pair_input
        )
        loader = DataLoader(ds, batch_size=8, shuffle=False)

        # Document is positive if ANY chunk scores positive
        doc_pred = 0
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out  = model(ids, mask)
            preds = out["det_logits"].argmax(dim=-1)
            if preds.any():
                doc_pred = 1
                break

        for key in ("attack_type", "attack_category", "source"):
            val = rec.get(key)
            if val is not None:
                buckets[key][str(val)].append((label, doc_pred))

    report: Dict[str, Any] = {}
    for key, groups in buckets.items():
        report[key] = {}
        for grp, pairs in groups.items():
            tp = sum(1 for l, p in pairs if p == 1)
            fn = sum(1 for l, p in pairs if p == 0)
            report[key][grp] = {
                "support": len(pairs),
                "recall":  round(tp / max(len(pairs), 1), 4),
                "tp": tp, "fn": fn,
            }
    return report


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def run_training(
    mode: str = "pair",           # "pair" | "chunk_only"
    augment: bool = True,
    backbone: str = "microsoft/deberta-v3-xsmall",
    epochs: int = 3,
    lr: float = 2e-5,
    batch_size: int = 8,
    lambda_tok: float = 1.0,
    seed: int = 42,
    output_dir: str = "checkpoints/joint-pair-deberta-v3-xsmall",
    max_length: int = 256,
    stride: int = 64,
) -> Dict[str, Any]:
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pair_input = (mode == "pair")

    print(f"\n{'='*60}")
    print(f"P3 Joint Training | mode={mode} | augment={augment} | seed={seed}")
    print(f"backbone={backbone} | epochs={epochs} | lr={lr} | device={device}")
    print(f"{'='*60}\n")

    tokenizer = AutoTokenizer.from_pretrained(backbone)

    train_records, _ = load_dataset_split("train")
    val_records,   _ = load_dataset_split("val")

    if augment:
        print(f"Applying obfuscation augmentation to {sum(r.get('label',0) for r in train_records)} injection records ...")
        train_records = augment_records(train_records, p=0.5)

    print(f"Building datasets (pair_input={pair_input}) ...")
    train_ds = PairInjectionDataset(train_records, tokenizer, max_length, stride, pair_input)
    val_ds   = PairInjectionDataset(val_records,   tokenizer, max_length, stride, pair_input)
    print(f"  train windows: {len(train_ds)}  |  val windows: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    model = SudarshanaJointModel(backbone=backbone, lambda_tok=lambda_tok)
    model.to(device)

    total_steps  = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    optimizer    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    history = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        ep0       = time.time()
        tr_loss   = _train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss  = _eval_epoch(model, val_loader, device)
        elapsed   = time.time() - ep0
        history.append({
            "epoch": epoch, "train_loss": round(tr_loss, 4),
            "val_loss": round(val_loss, 4), "time_sec": round(elapsed, 1),
        })
        print(f"Epoch {epoch}/{epochs} — train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  ({elapsed:.0f}s)")

    total_time = time.time() - t0

    # Recall-by-category evaluation on val set
    print("\nEvaluating recall by category on val set ...")
    recall_report = evaluate_recall_by_category(
        model, tokenizer, val_records, device, pair_input, max_length, stride
    )

    # Freeze and save
    model.freeze()
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

    metadata = {
        "mode":           mode,
        "pair_input":     pair_input,
        "augment":        augment,
        "backbone":       backbone,
        "epochs":         epochs,
        "learning_rate":  lr,
        "batch_size":     batch_size,
        "lambda_tok":     lambda_tok,
        "seed":           seed,
        "max_length":     max_length,
        "stride":         stride,
        "total_time_sec": round(total_time, 1),
        "history":        history,
        "recall_by_category": recall_report,
        "timestamp":      time.strftime("%Y-%m-%d %H:%M:%S"),
        "frozen":         True,
    }
    with open(os.path.join(output_dir, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nFrozen checkpoint saved → {output_dir}")
    return metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="P3 Joint Training Pipeline")
    p.add_argument("--mode",       default="pair",    choices=["pair", "chunk_only"])
    p.add_argument("--no_augment", action="store_true")
    p.add_argument("--backbone",   default="microsoft/deberta-v3-xsmall")
    p.add_argument("--epochs",     type=int,   default=3)
    p.add_argument("--lr",         type=float, default=2e-5)
    p.add_argument("--batch_size", type=int,   default=8)
    p.add_argument("--lambda_tok", type=float, default=1.0)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--output_dir", default="checkpoints/joint-pair-deberta-v3-xsmall")
    args = p.parse_args()

    run_training(
        mode=args.mode,
        augment=not args.no_augment,
        backbone=args.backbone,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        lambda_tok=args.lambda_tok,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
