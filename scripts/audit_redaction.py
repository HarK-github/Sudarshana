"""Audit Script for Sudarshana Token Head and Redaction Engine.
Evaluates the updated structure-aware boundary expansion, gap-filling, and calibrated 50% fallback ratio
against the baseline PySBD 35% configuration on test.jsonl (n=206 total, 125 attacked).
Computes before/after comparison with 95% Wilson intervals and 95% Bootstrap intervals.
"""

import os
import json
import time
import math
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
from transformers import AutoTokenizer

from sudarshana.models.token_head import SudarshanaTokenClassifier
from sudarshana.redaction.redact import (
    redact,
    expand_spans_to_sentences,
    BLOCKED_SENTINEL,
    predict_spans,
    set_inference_model,
    DEFAULT_FALLBACK_RATIO
)
from sudarshana.evaluation.evaluate import (
    predict_document_windows,
    extract_spans_from_record
)
from sudarshana.evaluation.metrics import (
    compute_span_f1,
    compute_character_coverage
)

# -----------------------------------------------------------------------------
# Statistical Helpers
# -----------------------------------------------------------------------------

def wilson_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    z = 1.959964
    denom = 1 + (z ** 2) / n
    center = (p + (z ** 2) / (2 * n)) / denom
    margin = (z / denom) * math.sqrt((p * (1 - p) / n) + ((z ** 2) / (4 * (n ** 2))))
    return (round(p, 4), round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4))


def bootstrap_interval(data: List[float], n_resamples: int = 2000, seed: int = 42) -> Tuple[float, float, float]:
    if not data:
        return (0.0, 0.0, 0.0)
    arr = np.array(data, dtype=np.float64)
    mean_val = float(np.mean(arr))
    if len(arr) == 1 or np.all(arr == arr[0]):
        return (round(mean_val, 4), round(mean_val, 4), round(mean_val, 4))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(arr), size=(n_resamples, len(arr)))
    resampled_means = np.mean(arr[indices], axis=1)
    lower = float(np.percentile(resampled_means, 2.5))
    upper = float(np.percentile(resampled_means, 97.5))
    return (round(mean_val, 4), round(lower, 4), round(upper, 4))


def run_v2_audit():
    print("=== STARTING P4 V2 AUDIT (STRUCTURE-AWARE & VAL-CALIBRATED) ===")
    checkpoint_dir = "checkpoints/prototype-token-head-deberta-v3-xsmall-20261110"
    backbone_name = "microsoft/deberta-v3-xsmall"
    tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    model = SudarshanaTokenClassifier(backbone_name_or_path=backbone_name, num_labels=3)
    weights_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    set_inference_model(model, tokenizer, threshold=0.35)

    with open("data/splits/test.jsonl") as f:
        test_records = [json.loads(line) for line in f]

    print(f"Loaded {len(test_records)} test records.")

    raw_v2_records = []
    
    # Trackers for V2 configuration (Structure-aware, gap=2, fallback=0.50)
    v2_blocked = 0
    v2_selective = 0
    v2_total_attacked = 0
    v2_total_injected_chars = 0
    v2_total_residual_chars = 0
    v2_selective_residual_chars = 0
    v2_selective_inj_chars = 0
    v2_samples_with_any_residual = 0
    v2_selective_samples_with_residual = 0
    
    v2_legit_retained_list = []
    v2_removed_to_inj_ratios = []
    v2_share_removed_non_inj = []
    v2_gold_coverage_list = []

    # Bucket trackers
    v2_buckets = {
        "<500": {"blocked": 0, "selective": 0, "n": 0, "retained": []},
        "500-1500": {"blocked": 0, "selective": 0, "n": 0, "retained": []},
        ">1500": {"blocked": 0, "selective": 0, "n": 0, "retained": []}
    }

    # Leaked character taxonomy tracker
    whitespace_leak_count = 0
    substantive_leak_count = 0
    leaked_character_details = []

    all_pred_spans = []
    all_gold_spans = []

    for idx, r in enumerate(test_records):
        doc = r["document"]
        golds = extract_spans_from_record(r)
        is_attacked = (len(golds) > 0)
        doc_len = len(doc)
        domain = r.get("domain", "text")

        # Gap-bridged prediction
        pred_spans = predict_spans(doc)
        all_pred_spans.append(pred_spans)
        all_gold_spans.append(golds)

        # Structure-aware expansion
        exp_spans = expand_spans_to_sentences(doc, pred_spans)

        # Redact with calibrated 0.50 fallback
        redacted_doc = redact(doc, pred_spans, fallback_ratio=DEFAULT_FALLBACK_RATIO)
        is_blocked = (redacted_doc == BLOCKED_SENTINEL)

        cov_res = compute_character_coverage(doc, golds, exp_spans, is_blocked)
        inj_len = sum(e - s for s, e in golds)

        if is_attacked:
            v2_total_attacked += 1
            v2_total_injected_chars += inj_len
            v2_gold_coverage_list.append(cov_res["gold_coverage"])

            if is_blocked:
                v2_blocked += 1
                v2_legit_retained_list.append(0.0)
                final_rem_len = doc_len
            else:
                v2_selective += 1
                final_rem_len = doc_len - len(redacted_doc)
                v2_legit_retained_list.append(cov_res["legitimate_retained"])
                
                # Residual on selective
                surv_inj_chars = int(round(cov_res["residual_payload"] * inj_len))
                v2_total_residual_chars += surv_inj_chars
                v2_selective_residual_chars += surv_inj_chars
                v2_selective_inj_chars += inj_len

                if surv_inj_chars > 0:
                    v2_samples_with_any_residual += 1
                    v2_selective_samples_with_residual += 1
                    
                    # Analyze leaked characters: are they whitespace/delimiter or words?
                    # Find unexcised characters
                    surv_chars = []
                    for g_s, g_e in golds:
                        for ci in range(g_s, g_e):
                            if not any(e_s <= ci < e_e for e_s, e_e in exp_spans):
                                surv_chars.append(doc[ci])
                    surv_str = "".join(surv_chars)
                    if surv_str.strip() == "":
                        whitespace_leak_count += 1
                    else:
                        substantive_leak_count += 1
                    leaked_character_details.append({
                        "index": idx,
                        "surviving_string": surv_str,
                        "is_whitespace_only": (surv_str.strip() == ""),
                        "surviving_count": len(surv_str)
                    })

            # Cuts ratio
            ratio_rem = (final_rem_len / max(inj_len, 1))
            v2_removed_to_inj_ratios.append(ratio_rem)
            
            non_inj_removed = max(0, final_rem_len - (inj_len - int(round(cov_res["residual_payload"] * inj_len))))
            share_non_inj = non_inj_removed / max(final_rem_len, 1)
            v2_share_removed_non_inj.append(share_non_inj)

            # Length bucket
            b_key = "<500" if doc_len < 500 else ("500-1500" if doc_len <= 1500 else ">1500")
            v2_buckets[b_key]["n"] += 1
            v2_buckets[b_key]["retained"].append(0.0 if is_blocked else cov_res["legitimate_retained"])
            if is_blocked:
                v2_buckets[b_key]["blocked"] += 1
            else:
                v2_buckets[b_key]["selective"] += 1

        raw_v2_records.append({
            "index": idx,
            "is_attacked": is_attacked,
            "domain": domain,
            "doc_length": doc_len,
            "golds": golds,
            "pred_spans": pred_spans,
            "expanded_spans": exp_spans,
            "is_blocked": is_blocked,
            "gold_coverage": cov_res["gold_coverage"],
            "residual_payload": cov_res["residual_payload"],
            "legitimate_retained": cov_res["legitimate_retained"]
        })

    # Save V2 raw outputs
    raw_v2_path = "reports/audit/raw_test_predictions.jsonl"
    with open(raw_v2_path, "w") as f:
        for r in raw_v2_records:
            f.write(json.dumps(r) + "\n")
    print(f"Saved audit predictions to: {raw_v2_path}")

    # -------------------------------------------------------------------------
    # Negative Control (Clean Counterfactuals)
    # -------------------------------------------------------------------------
    clean_counterfactuals = []
    for r in test_records:
        golds = extract_spans_from_record(r)
        if not golds:
            continue
        doc = r["document"]
        s, e = golds[0]
        clean_doc = (doc[:s] + doc[e:]).strip()
        clean_counterfactuals.append(clean_doc)
    
    cf_touched = 0
    cf_blocked = 0
    cf_retained = []
    for c_doc in clean_counterfactuals:
        p_spans = predict_spans(c_doc)
        red_c = redact(c_doc, p_spans, fallback_ratio=DEFAULT_FALLBACK_RATIO)
        is_blk = (red_c == BLOCKED_SENTINEL)
        if len(p_spans) > 0:
            cf_touched += 1
        if is_blk:
            cf_blocked += 1
            cf_retained.append(0.0)
        else:
            cf_retained.append(len(red_c) / max(len(c_doc), 1))
    
    w_cf_touch = wilson_interval(cf_touched, len(clean_counterfactuals))
    w_cf_block = wilson_interval(cf_blocked, len(clean_counterfactuals))
    cf_ret_ci = bootstrap_interval(cf_retained)

    # -------------------------------------------------------------------------
    # PRINT BEFORE VS AFTER SCORECARD
    # -------------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("=== P4 BEFORE VS AFTER AUDIT SCORECARD (HELD-OUT TEST SET) ===")
    print("=" * 65)
    
    w_blk = wilson_interval(v2_blocked, v2_total_attacked)
    w_sel = wilson_interval(v2_selective, v2_total_attacked)
    ret_ci = bootstrap_interval(v2_legit_retained_list)
    cov_ci = bootstrap_interval(v2_gold_coverage_list)
    
    v2_res_pct = (v2_total_residual_chars / max(v2_total_injected_chars, 1)) * 100
    v2_sel_res_pct = (v2_selective_residual_chars / max(v2_selective_inj_chars, 1)) * 100
    w_res_share = wilson_interval(v2_samples_with_any_residual, v2_total_attacked)
    w_sel_res_share = wilson_interval(v2_selective_samples_with_residual, v2_selective)

    # 1-to-1 Span F1
    span_res = compute_span_f1(all_pred_spans, all_gold_spans, exact_match=False)

    print(f"{'Metric':<35} | {'Before (PySBD 35%)':<22} | {'After (Structure 50%)':<25}")
    print("-" * 88)
    print(f"{'Fallback Block Rate':<35} | 56.80% [48.0%, 65.1%]    | {w_blk[0]*100:.2f}% [{w_blk[1]*100:.1f}%, {w_blk[2]*100:.1f}%]")
    print(f"{'Selective Redaction Rate':<35} | 43.20% [34.9%, 52.0%]    | {w_sel[0]*100:.2f}% [{w_sel[1]*100:.1f}%, {w_sel[2]*100:.1f}%]")
    print(f"{'Legitimate Content Retained':<35} | 38.55% [30.8%, 46.1%]    | {ret_ci[0]*100:.2f}% [{ret_ci[1]*100:.1f}%, {ret_ci[2]*100:.1f}%]")
    print(f"{'Gold Span Character Coverage':<35} | 98.51% [97.8%, 99.2%]    | {cov_ci[0]*100:.2f}% [{cov_ci[1]*100:.1f}%, {cov_ci[2]*100:.1f}%]")
    print(f"{'Residual Payload Rate (Overall)':<35} | 1.49% (310 chars)        | {v2_res_pct:.2f}% ({v2_total_residual_chars} chars)")
    print(f"{'Residual Payload (Selective Only)':<35} | 3.44% (310 / 9,006 chars)| {v2_sel_res_pct:.2f}% ({v2_selective_residual_chars} / {v2_selective_inj_chars} chars)")
    print(f"{'Samples Leaking Any Residual':<35} | 17.60% (22/125)          | {w_res_share[0]*100:.2f}% ({v2_samples_with_any_residual}/125)")
    print(f"{'Span Overlap Recall (1-to-1)':<35} | 99.20% (124/125)         | {span_res['span_overlap_recall']*100:.2f}%")
    print(f"{'Span Overlap F1 (1-to-1)':<35} | 34.42%                  | {span_res['span_overlap_f1']*100:.2f}%")
    
    print("\n--- DECOMPOSITION BY DOCUMENT LENGTH BUCKET (AFTER) ---")
    for b_key in ["<500", "500-1500", ">1500"]:
        b = v2_buckets[b_key]
        n_b = b["n"]
        w_b_blk = wilson_interval(b["blocked"], n_b)
        w_b_sel = wilson_interval(b["selective"], n_b)
        b_ret_ci = bootstrap_interval(b["retained"])
        print(f"Bucket {b_key:<9} (n={n_b:<2}): Block: {w_b_blk[0]*100:5.1f}% [{w_b_blk[1]*100:4.1f}-{w_b_blk[2]*100:4.1f}%] | Selective: {w_b_sel[0]*100:5.1f}% [{w_b_sel[1]*100:4.1f}-{w_b_sel[2]*100:4.1f}%] | Retained: {b_ret_ci[0]*100:5.1f}% [{b_ret_ci[1]*100:4.1f}-{b_ret_ci[2]*100:4.1f}%]")

    print("\n--- CLEAN COUNTERFACTUAL CONTROLS (n=125) ---")
    print(f"Documents Touched: {cf_touched}/{len(clean_counterfactuals)} = {w_cf_touch[0]*100:.2f}% (Wilson: [{w_cf_touch[1]*100:.2f}%, {w_cf_touch[2]*100:.2f}%])")
    print(f"Documents Blocked: {cf_blocked}/{len(clean_counterfactuals)} = {w_cf_block[0]*100:.2f}% (Wilson: [{w_cf_block[1]*100:.2f}%, {w_cf_block[2]*100:.2f}%])")
    print(f"Legitimate Content Preserved: Mean {cf_ret_ci[0]*100:.2f}% (Bootstrap: [{cf_ret_ci[1]*100:.2f}%, {cf_ret_ci[2]*100:.2f}%])")

    print("\n--- LEAKED RESIDUAL TAXONOMY ---")
    print(f"Total leaking samples: {len(leaked_character_details)}")
    print(f" - Whitespace-only boundary leaks ('\\n', ' '): {whitespace_leak_count} samples")
    print(f" - Substantive character leaks: {substantive_leak_count} samples")
    for item in leaked_character_details:
        print(f"   Sample {item['index']:>3}: count={item['surviving_count']} chars, whitespace_only={item['is_whitespace_only']}, text={repr(item['surviving_string'][:40])}")

if __name__ == "__main__":
    run_v2_audit()

