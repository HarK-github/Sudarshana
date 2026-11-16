"""Evaluation script for Table 3.5: Obfuscation Robustness.
Compares Baseline (Step 1) vs. With Normalizer (Step 2) vs. With Normalizer & Augmentation (Step 6).
Conforms strictly to Report Page 10, Section 3.5.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import torch

# Ensure Sudarshana package is importable
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.data.dataset import PromptInjectionDataset, PromptInjectionRecord
from src.data.normalizer import TextNormalizer
from src.data.collators import SingleStringCollator, PairInputCollator, get_tokenizer
from src.data.augmentations import (
    apply_leetspeak,
    apply_homoglyphs,
    apply_zero_width,
    apply_base64_wrapper,
    apply_markdown_embed,
)
from src.models.deberta_multitask import DebertaMultiTaskFilter


def create_perturbed_dataset(
    records: List[PromptInjectionRecord],
    perturb_fn: Any,
    seed: int = 42,
) -> List[PromptInjectionRecord]:
    rng = random.Random(seed)
    perturbed: List[PromptInjectionRecord] = []
    for r in records:
        if perturb_fn is None:
            perturbed.append(r)
            continue
        p_text, p_spans = perturb_fn(r.text, r.injection_char_spans, rng=rng)
        perturbed.append(
            PromptInjectionRecord(
                sample_id=r.sample_id,
                doc_id=r.doc_id,
                user_task=r.user_task,
                text=p_text,
                label=r.label,
                injection_char_spans=p_spans,
                attack_type_id=r.attack_type_id,
                attack_category=r.attack_category,
                position_bucket=r.position_bucket,
                domain=r.domain,
            )
        )
    return perturbed


def evaluate_model(
    model: DebertaMultiTaskFilter,
    records: List[PromptInjectionRecord],
    tokenizer: Any,
    device: torch.device,
    tau: float,
    use_normalizer: bool,
    pair_input: bool,
    max_length: int = 512,
    batch_size: int = 32,
) -> Dict[str, float]:
    normalizer = TextNormalizer() if use_normalizer else None
    model.eval()

    if pair_input:
        collator = PairInputCollator(
            tokenizer=tokenizer,
            max_task_length=64,
            max_chunk_length=256,
            total_max_length=323,
        )
    else:
        collator = SingleStringCollator(
            tokenizer=tokenizer,
            max_length=max_length,
        )

    prepared_dicts = []
    y_true = []
    for r in records:
        text = r.text
        task = r.user_task
        spans = r.injection_char_spans
        if normalizer is not None:
            text, spans = normalizer.normalize(text, spans)
            task, _ = normalizer.normalize(task, [])

        rec_dict = PromptInjectionRecord(
            sample_id=r.sample_id,
            doc_id=r.doc_id,
            user_task=task,
            text=text,
            label=r.label,
            injection_char_spans=spans,
            attack_type_id=r.attack_type_id,
            attack_category=r.attack_category,
            position_bucket=r.position_bucket,
            domain=r.domain,
        ).to_dict()
        prepared_dicts.append(rec_dict)
        y_true.append(r.label)

    y_pred = []
    with torch.no_grad():
        for i in range(0, len(prepared_dicts), batch_size):
            chunk_batch = prepared_dicts[i : i + batch_size]
            batch = collator(chunk_batch)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = out["chunk_probs"].cpu().tolist()
            for p in probs:
                y_pred.append(1 if p >= tau else 0)

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)

    recall = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)
    prec = tp / max(1, tp + fp)
    f1 = 2 * prec * recall / max(1e-6, prec + recall)

    return {
        "recall": recall,
        "fpr": fpr,
        "precision": prec,
        "f1": f1,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Table 3.5 Evaluation on device: {device.type.upper()}")

    test_ds = PromptInjectionDataset(repo_root / "data/splits/test.jsonl")
    records = test_ds.get_records()

    tokenizer = get_tokenizer(repo_root / "checkpoints/prototype-token-head-deberta-v3-xsmall-seed42")

    # Load thresholds
    tau_s1 = json.load(open(repo_root / "models/final_step1_baseline/calibrated_threshold.json"))["calibrated_tau"]
    tau_s2 = json.load(open(repo_root / "models/final_step2_normalizer/calibrated_threshold.json"))["calibrated_tau"]
    tau_s6 = json.load(open(repo_root / "models/final_step6_augmented/calibrated_threshold.json"))["calibrated_tau"]

    print(f"Loaded thresholds: Step 1 = {tau_s1:.4f}, Step 2 = {tau_s2:.4f}, Step 6 = {tau_s6:.4f}")

    # Load models
    m1 = DebertaMultiTaskFilter.from_pretrained(repo_root / "checkpoints/step1_baseline/seed_42", device=device)
    m2 = DebertaMultiTaskFilter.from_pretrained(repo_root / "checkpoints/step2_normalizer/seed_42", device=device)
    m6 = DebertaMultiTaskFilter.from_pretrained(repo_root / "checkpoints/step6_augmented/seed_42", device=device)

    perturbations = [
        ("Base64 wrapper", apply_base64_wrapper),
        ("Homoglyphs", apply_homoglyphs),
        ("Zero-width characters", apply_zero_width),
        ("Leetspeak", apply_leetspeak),
        ("Markdown embeds", apply_markdown_embed),
    ]

    results_table = []
    full_dict = {}

    for name, p_fn in perturbations:
        p_records = create_perturbed_dataset(records, p_fn, seed=42)

        res_s1 = evaluate_model(m1, p_records, tokenizer, device, tau=tau_s1, use_normalizer=False, pair_input=False)
        res_s2 = evaluate_model(m2, p_records, tokenizer, device, tau=tau_s2, use_normalizer=True, pair_input=False)
        res_s6 = evaluate_model(m6, p_records, tokenizer, device, tau=tau_s6, use_normalizer=True, pair_input=True)

        results_table.append({
            "obfuscation_type": name,
            "baseline_recall": res_s1["recall"],
            "with_normalizer_recall": res_s2["recall"],
            "with_normalizer_and_aug_recall": res_s6["recall"],
            "step6_metrics": res_s6,
        })
        full_dict[name] = {
            "baseline_recall": res_s1["recall"],
            "with_normalizer_recall": res_s2["recall"],
            "with_normalizer_and_aug_recall": res_s6["recall"],
            "step1": res_s1,
            "step2": res_s2,
            "step6": res_s6,
        }

        print(f"{name:<24} | Step 1: {res_s1['recall']:6.2%} | Step 2: {res_s2['recall']:6.2%} | Step 6: {res_s6['recall']:6.2%}")

    out_path = repo_root / "results/table_3_5_obfuscation_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_dict, f, indent=2)

    print(f"\nSaved Table 3.5 comparison to: {out_path}")


if __name__ == "__main__":
    main()
