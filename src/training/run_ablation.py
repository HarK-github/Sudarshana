"""CLI entry point for executing all 6 rows of the Ablation Ladder (Section 7)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import yaml
import torch

from ..data.dataset import PromptInjectionDataset, PromptInjectionRecord
from ..data.collators import SingleStringCollator, PairInputCollator, get_tokenizer
from ..models.deberta_multitask import DebertaMultiTaskFilter, DebertaMultiTaskConfig
from ..models.reference_detector import OffTheShelfReferenceDetector
from .trainer import UnifiedTrainer, TrainingArguments
from .metrics_hooks import evaluate_at_fixed_recall


def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Loads a YAML configuration, recursively merging '_base_' configurations if present."""
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if "_base_" in data:
        base_path = cfg_path.parent / data["_base_"]
        if not base_path.exists():
            base_path = Path(data["_base_"])
        base_data = load_yaml_config(base_path)
        base_data.update(data)
        data = base_data

    return data


def format_metrics_table(step_name: str, metrics: Dict[str, Any]) -> str:
    """Formats stratified evaluation metrics matching Report Tables 3.1 - 3.7."""
    lines = [
        f"\n=======================================================",
        f"  Ablation Ladder Results: {step_name.upper()}",
        f"=======================================================",
    ]
    ov = metrics.get("overall", {})
    lines.append(f"Overall Recall:     {ov.get('recall', 0.0):.4f} ({ov.get('tp', 0)}/{ov.get('tp', 0) + ov.get('fn', 0)})")
    lines.append(f"Overall FPR:        {ov.get('fpr', 0.0):.4f} ({ov.get('fp', 0)}/{ov.get('fp', 0) + ov.get('tn', 0)})")
    lines.append(f"Overall F1:         {ov.get('f1', 0.0):.4f}")
    lines.append(f"Precision:          {ov.get('precision', 0.0):.4f}")

    # Table 3.3: Recall by Position
    by_pos = metrics.get("recall_by_position", {})
    lines.append("\n[Table 3.3] Recall by Position Bucket:")
    for pos in ["start", "middle", "end"]:
        rec = by_pos.get(pos, {}).get("recall", 0.0)
        sup = by_pos.get(pos, {}).get("support", 0)
        lines.append(f"  - {pos.capitalize():<8}: {rec:.4f} (support: {sup})")

    # Table 3.4: Recall by Attack Category
    by_cat = metrics.get("recall_by_category", {})
    lines.append("\n[Table 3.4] Recall by Attack Category:")
    for cat in ["task_irrelevant", "task_relevant", "targeted"]:
        rec = by_cat.get(cat, {}).get("recall", 0.0)
        sup = by_cat.get(cat, {}).get("support", 0)
        lines.append(f"  - {cat:<16}: {rec:.4f} (support: {sup})")

    # FPR on Hard Negatives vs Clean
    by_neg = metrics.get("fpr_by_negative_category", {})
    lines.append("\n[Section 4.3] False Positive Rate by Negative Category:")
    for cat in ["clean", "hard_negative"]:
        fpr_v = by_neg.get(cat, {}).get("fpr", 0.0)
        sup = by_neg.get(cat, {}).get("support", 0)
        lines.append(f"  - {cat:<16}: {fpr_v:.4f} (support: {sup})")

    # Table 3.7: Text QA vs Code QA
    by_dom = metrics.get("by_domain", {})
    lines.append("\n[Table 3.7] Modality Breakdown (Text QA vs Code QA):")
    for dom in ["text_qa", "code_qa"]:
        rec = by_dom.get(dom, {}).get("recall", 0.0)
        fpr_v = by_dom.get(dom, {}).get("fpr", 0.0)
        sup = by_dom.get(dom, {}).get("support", 0)
        lines.append(f"  - {dom:<10}: Recall={rec:.4f}, FPR={fpr_v:.4f} (support: {sup})")

    lines.append("=======================================================\n")
    return "\n".join(lines)


def run_ablation(
    config: Dict[str, Any],
    seed: Optional[int] = None,
    epochs: Optional[int] = None,
    eval_only: bool = False,
    reference_detector: bool = False,
) -> Dict[str, Any]:
    step_name = config.get("step_name", "step1_baseline")
    chosen_seed = seed if seed is not None else config.get("default_seed", 42)
    chosen_epochs = epochs if epochs is not None else config.get("epochs", 3)

    data_dir = Path(config.get("data_dir", "data/splits"))
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"
    test_path = data_dir / "test.jsonl"

    # Off-the-shelf reference detector evaluation (Table 3.9 / Open Item #4)
    if reference_detector:
        print(f"Running zero-shot secondary reference detector...")
        ref_detector = OffTheShelfReferenceDetector()
        test_ds = PromptInjectionDataset(test_path)
        eval_res = ref_detector.evaluate(test_ds.get_records(), threshold=0.5)
        print(format_metrics_table("Reference_Detector", eval_res["metrics"]))
        return eval_res

    # Select Tokenizer
    backbone = config.get("backbone", "microsoft/deberta-v3-base")
    fallback = config.get("fallback_backbone", "microsoft/deberta-v3-xsmall")
    try:
        tokenizer = get_tokenizer(backbone)
    except Exception:
        tokenizer = get_tokenizer(fallback)
        backbone = fallback

    # Select Collator based on pair_input flag
    pair_input = config.get("pair_input", False)
    if pair_input:
        collator = PairInputCollator(
            tokenizer=tokenizer,
            max_task_length=config.get("max_task_length", 64),
            max_chunk_length=config.get("max_chunk_length", 256),
            total_max_length=config.get("total_max_length", 323),
        )
    else:
        collator = SingleStringCollator(
            tokenizer=tokenizer,
            max_length=config.get("max_tokens", 512),
        )

    # Ingest Datasets
    print(f"Loading datasets from {data_dir}...")
    raw_train_ds = PromptInjectionDataset(train_path) if train_path.exists() and not eval_only else None
    raw_val_ds = PromptInjectionDataset(val_path) if val_path.exists() else None
    raw_test_ds = PromptInjectionDataset(test_path) if test_path.exists() else None

    if raw_test_ds is None:
        raise FileNotFoundError(f"Test split not found at {test_path}")

    train_records = raw_train_ds.get_records() if raw_train_ds is not None else []
    val_records = raw_val_ds.get_records() if raw_val_ds is not None else []
    test_records = raw_test_ds.get_records()

    # Step 4 / H3: Counterfactual pairs
    if config.get("counterfactual_pairs", False) and train_records:
        from src.data.counterfactuals import generate_counterfactual_pairs
        train_records = generate_counterfactual_pairs(train_records, seed=chosen_seed)

    # Step 6 / H2: Obfuscation Augmentation
    if config.get("obf_augment", False) and train_records:
        from src.data.augmentations import generate_obfuscated_training_set
        train_records = generate_obfuscated_training_set(train_records, augment_ratio=0.5, seed=chosen_seed)

    # Step 2+ / H2: Text Normalization with Span Re-anchoring
    if config.get("normalize", False):
        from src.data.normalizer import TextNormalizer
        normalizer = TextNormalizer(nfkc=True, strip_zero_width=True, decode_b64=True, decode_hex=True)

        def _norm_recs(recs):
            normed = []
            for r in recs:
                norm_text, norm_spans = normalizer.normalize(r.text, r.injection_char_spans)
                norm_task, _ = normalizer.normalize(r.user_task, [])
                normed.append(PromptInjectionRecord(
                    sample_id=r.sample_id,
                    doc_id=r.doc_id,
                    user_task=norm_task,
                    text=norm_text,
                    label=r.label,
                    injection_char_spans=norm_spans,
                    attack_type_id=r.attack_type_id,
                    attack_category=r.attack_category,
                    position_bucket=r.position_bucket,
                    domain=r.domain,
                    raw_metadata=r.raw_metadata,
                ))
            return normed

        if train_records:
            train_records = _norm_recs(train_records)
        if val_records:
            val_records = _norm_recs(val_records)
        test_records = _norm_recs(test_records)
    else:
        def _norm_recs(recs):
            return recs

    chunk_win = (
        (config.get("chunk_size", 256), config.get("chunk_stride", 192))
        if config.get("chunking", False)
        else None
    )
    train_ds = PromptInjectionDataset(train_records, chunk_window=chunk_win, tokenizer=tokenizer) if train_records else None
    val_ds = PromptInjectionDataset(val_records, chunk_window=chunk_win, tokenizer=tokenizer) if val_records else None
    test_ds = PromptInjectionDataset(test_records, chunk_window=chunk_win, tokenizer=tokenizer)

    # Setup Trainer Output Directory
    out_dir = Path(config.get("output_dir", f"checkpoints/{step_name}")) / f"seed_{chosen_seed}"

    # Build Model or Load Checkpoint
    if eval_only and (out_dir / "model_weights.pt").exists():
        print(f"Loading trained checkpoint from {out_dir} for evaluation...")
        model = DebertaMultiTaskFilter.from_pretrained(out_dir)
    else:
        model_cfg = DebertaMultiTaskConfig(
            backbone=backbone,
            use_token_head=config.get("use_token_head", False),
            use_type_head=config.get("use_type_head", False),
            lambda_tok=config.get("lambda_tok", 0.25),
            lambda_type=config.get("lambda_type", 0.1),
        )
        model = DebertaMultiTaskFilter(config=model_cfg)
    args = TrainingArguments(
        output_dir=str(out_dir),
        seed=chosen_seed,
        epochs=chosen_epochs,
        per_device_batch_size=config.get("per_device_batch_size", 16),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 2),
        backbone_lr=float(config.get("backbone_lr", 3e-5)),
        heads_lr=float(config.get("heads_lr", 1.5e-4)),
        weight_decay=float(config.get("weight_decay", 0.01)),
        warmup_ratio=float(config.get("warmup_ratio", 0.1)),
        max_grad_norm=float(config.get("max_grad_norm", 1.0)),
        fp16=config.get("fp16", True),
        step_name=step_name,
        pooling_method=config.get("pooling_method", "top2_damped" if config.get("chunking", False) else "single"),
        top_k=config.get("top_k", 2),
        use_token_head=config.get("use_token_head", False),
        use_type_head=config.get("use_type_head", False),
    )

    trainer = UnifiedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        collate_fn=collator,
    )

    # Execute Training
    if not eval_only and train_ds is not None:
        print(f"\nTraining {step_name} with seed={chosen_seed} for {chosen_epochs} epochs...")
        trainer.train()
        if val_ds is not None:
            print(f"Exporting final validation predictions for {step_name}...")
            trainer.evaluate(val_ds, split_name="val", export_artifacts=True)

    # Execute Test Evaluation
    print(f"\nEvaluating {step_name} on held-out test split...")
    eval_result = trainer.evaluate(test_ds, split_name="test", export_artifacts=True)
    metrics = eval_result["metrics"]

    # Calculate Fixed Recall Metric (Table 3.2: Hard Negative FPR at fixed recall)
    y_true = [int(p["label"]) for p in eval_result["doc_predictions"]]
    y_prob = [float(p["doc_prob"]) for p in eval_result["doc_predictions"]]
    fixed_rec_metrics = evaluate_at_fixed_recall(y_true, y_prob, target_recall=0.90)
    metrics["hard_neg_at_90_recall"] = fixed_rec_metrics

    # Print Report Tables
    table_str = format_metrics_table(step_name, metrics)
    print(table_str)

    # Generate Table 3.5 Obfuscation Robustness Suite if obf_augment is enabled
    if config.get("obf_augment", False):
        print(f"\n=======================================================")
        print(f"  [Table 3.5] Obfuscation Robustness Evaluation")
        print(f"=======================================================")
        from src.data.augmentations import generate_obfuscation_eval_splits
        obf_splits = generate_obfuscation_eval_splits(raw_test_ds.get_records(), seed=chosen_seed)
        table_3_5_results = {}
        for obf_name, obf_recs in obf_splits.items():
            if config.get("normalize", False):
                obf_recs = _norm_recs(obf_recs)
            obf_ds = PromptInjectionDataset(obf_recs)
            obf_eval = trainer.evaluate(obf_ds, split_name=f"obf_{obf_name}", export_artifacts=False)
            table_3_5_results[obf_name] = {
                "recall": obf_eval["metrics"]["overall"]["recall"],
                "fpr": obf_eval["metrics"]["overall"]["fpr"],
                "f1": obf_eval["metrics"]["overall"]["f1"],
                "precision": obf_eval["metrics"]["overall"]["precision"],
            }
            print(f"  - {obf_name:15s}: Recall={table_3_5_results[obf_name]['recall']:.4f}, FPR={table_3_5_results[obf_name]['fpr']:.4f}, F1={table_3_5_results[obf_name]['f1']:.4f}")
        print(f"=======================================================\n")
        metrics["table_3_5_obfuscation"] = table_3_5_results

    # Save results summary JSON
    results_dir = Path(config.get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"{step_name}_seed{chosen_seed}_metrics.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Results saved to: {results_file}")

    if config.get("obf_augment", False) and "table_3_5_obfuscation" in metrics:
        obf_results_file = results_dir / f"table_3_5_obfuscation_{step_name}_seed{chosen_seed}.json"
        with open(obf_results_file, "w", encoding="utf-8") as f:
            json.dump(metrics["table_3_5_obfuscation"], f, indent=2)

    return eval_result


def main():
    parser = argparse.ArgumentParser(description="Run Sudarshana Prompt Injection Ablation Ladder")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    parser.add_argument("--all-steps", action="store_true", help="Run all 6 ablation steps sequentially")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch count override")
    parser.add_argument("--eval_only", action="store_true", help="Skip training and run evaluation only")
    parser.add_argument("--reference_detector", action="store_true", help="Evaluate secondary reference detector")

    args = parser.parse_args()

    if args.all_steps:
        step_configs = [
            "configs/step1_baseline.yaml",
            "configs/step2_normalizer.yaml",
            "configs/step3_chunking.yaml",
            "configs/step4_pair_input.yaml",
            "configs/step5_token_head.yaml",
            "configs/step6_augmented.yaml",
        ]
        for cfg_path in step_configs:
            print(f"\n=======================================================")
            print(f"  EXECUTING ABLATION STEP: {cfg_path}")
            print(f"=======================================================\n")
            cfg = load_yaml_config(cfg_path)
            run_ablation(
                config=cfg,
                seed=args.seed,
                epochs=args.epochs,
                eval_only=args.eval_only,
                reference_detector=False,
            )
    elif args.config:
        config = load_yaml_config(args.config)
        run_ablation(
            config=config,
            seed=args.seed,
            epochs=args.epochs,
            eval_only=args.eval_only,
            reference_detector=args.reference_detector,
        )
    else:
        parser.error("Either --config or --all-steps must be specified.")


if __name__ == "__main__":
    main()

