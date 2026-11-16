"""Latency and throughput profiling for Table 3.8 (Task 6.2).

Measures and logs detector GPU/CPU inference latency per ~2k-token document:
- Baseline: Single-string 512-truncated encoder forward pass.
- Full Pipeline (Step 6): Text Normalizer + Sliding-window 256-token chunking (64 overlap)
  + Pair-Input collator + Joint detection & token classification heads + Max pooling.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import torch

from src.data.dataset import PromptInjectionRecord
from src.data.normalizer import TextNormalizer
from src.data.collators import SingleStringCollator, PairInputCollator, get_tokenizer
from src.models.deberta_multitask import DebertaMultiTaskFilter, DebertaMultiTaskConfig


def create_2k_token_document() -> PromptInjectionRecord:
    """Creates a realistic synthetic document of ~2,000 tokens."""
    paragraph = (
        "In modern enterprise cloud architectures, asynchronous distributed message queues "
        "facilitate decoupled microservice communication. Event-driven architectures rely on brokers "
        "such as Apache Kafka or RabbitMQ to stream telemetry, order state updates, and security logs. "
        "Each worker process consumes messages in parallel, committing offsets only upon successful "
        "downstream database persistence. System failovers and network partitions require exponential "
        "backoff and idempotent retry semantics to prevent duplicate transaction execution. "
    )
    # Repeat paragraph ~20 times to produce approx 2,000 subword tokens (~1,400 words)
    long_doc = "\n\n".join([f"Section {i}: {paragraph}" for i in range(1, 21)])
    user_task = "Summarize the fault tolerance mechanisms of the distributed queue."

    return PromptInjectionRecord(
        sample_id="bench_doc_2k",
        doc_id="bench_doc_2k",
        user_task=user_task,
        text=long_doc,
        label=0,
        injection_char_spans=[],
        attack_type_id=-100,
        attack_category="clean",
        position_bucket="none",
        domain="text_qa",
    )


def benchmark_baseline(
    model: DebertaMultiTaskFilter,
    tokenizer: Any,
    record: PromptInjectionRecord,
    device: torch.device,
    n_iters: int = 30,
) -> Dict[str, float]:
    """Profiles Step 1 Baseline (Single string 512 max truncation)."""
    collator = SingleStringCollator(tokenizer=tokenizer, max_length=512)
    batch = collator([record.to_dict()])
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    model.eval()
    # Warm-up
    with torch.no_grad():
        for _ in range(5):
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Timed runs
    latencies: List[float] = []
    with torch.no_grad():
        for _ in range(n_iters):
            t0 = time.perf_counter()
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    return {
        "mean_latency_ms": float(np.mean(latencies)),
        "median_latency_ms": float(np.median(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "std_latency_ms": float(np.std(latencies)),
        "throughput_docs_sec": float(1000.0 / np.mean(latencies)),
        "num_chunks_processed": 1,
    }


def benchmark_full_pipeline(
    model: DebertaMultiTaskFilter,
    tokenizer: Any,
    record: PromptInjectionRecord,
    device: torch.device,
    n_iters: int = 30,
) -> Dict[str, float]:
    """Profiles Step 6 Full Pipeline (Normalizer + Chunking + Pair Input + Joint Heads + Pooling)."""
    normalizer = TextNormalizer()
    collator = PairInputCollator(
        tokenizer=tokenizer,
        max_task_length=64,
        max_chunk_length=256,
        total_max_length=323,
    )

    # 1. Normalization
    t_norm_0 = time.perf_counter()
    norm_text, norm_spans = normalizer.normalize(record.text, record.injection_char_spans)
    norm_task, _ = normalizer.normalize(record.user_task, [])
    t_norm_1 = time.perf_counter()
    norm_latency_ms = (t_norm_1 - t_norm_0) * 1000.0

    # 2. Chunking (sliding window: 256 size, 192 stride)
    words = norm_text.split()
    chunk_size_words = 180  # approx 256 tokens
    stride_words = 135     # approx 192 stride
    chunk_records = []
    c_idx = 0
    start = 0
    while start < len(words):
        end = min(start + chunk_size_words, len(words))
        chunk_str = " ".join(words[start:end])
        chunk_records.append(
            PromptInjectionRecord(
                sample_id=f"{record.doc_id}_chunk_{c_idx:02d}",
                doc_id=record.doc_id,
                user_task=norm_task,
                text=chunk_str,
                label=record.label,
                injection_char_spans=[],
                attack_type_id=record.attack_type_id,
                attack_category=record.attack_category,
                position_bucket=record.position_bucket,
                domain=record.domain,
            ).to_dict()
        )
        c_idx += 1
        if end >= len(words):
            break
        start += stride_words

    num_chunks = len(chunk_records)
    batch = collator(chunk_records)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    model.eval()
    # Warm-up
    with torch.no_grad():
        for _ in range(5):
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Timed runs
    latencies: List[float] = []
    with torch.no_grad():
        for _ in range(n_iters):
            t0 = time.perf_counter()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Max pooling across chunks
            chunk_probs = outputs["chunk_probs"].cpu().numpy()
            _ = float(np.max(chunk_probs))
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0 + norm_latency_ms)

    return {
        "mean_latency_ms": float(np.mean(latencies)),
        "median_latency_ms": float(np.median(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "std_latency_ms": float(np.std(latencies)),
        "throughput_docs_sec": float(1000.0 / np.mean(latencies)),
        "num_chunks_processed": num_chunks,
        "norm_preprocessing_ms": float(norm_latency_ms),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Table 3.8 Latency Profiling on device: {device.type.upper()}")

    bench_doc = create_2k_token_document()
    tokenizer = get_tokenizer("checkpoints/prototype-token-head-deberta-v3-xsmall-seed42")

    # Load Baseline Model (Step 1)
    step1_dir = Path("checkpoints/step1_baseline/seed_42")
    if step1_dir.exists() and (step1_dir / "model_weights.pt").exists():
        baseline_model = DebertaMultiTaskFilter.from_pretrained(step1_dir, device=device)
    else:
        cfg = DebertaMultiTaskConfig(use_token_head=False, use_type_head=False)
        baseline_model = DebertaMultiTaskFilter(cfg).to(device)

    # Load Full Pipeline Model (Step 6)
    step6_dir = Path("checkpoints/step6_augmented/seed_42")
    if step6_dir.exists() and (step6_dir / "model_weights.pt").exists():
        step6_model = DebertaMultiTaskFilter.from_pretrained(step6_dir, device=device)
    else:
        cfg = DebertaMultiTaskConfig(use_token_head=True, use_type_head=False)
        step6_model = DebertaMultiTaskFilter(cfg).to(device)

    print("\nBenchmarking Step 1 (Baseline - 512 truncation)...")
    baseline_stats = benchmark_baseline(baseline_model, tokenizer, bench_doc, device=device, n_iters=30)
    print(f"  Baseline Latency: {baseline_stats['mean_latency_ms']:.2f} ms (Throughput: {baseline_stats['throughput_docs_sec']:.2f} docs/s)")

    print("\nBenchmarking Step 6 (Full Pipeline - Normalizer + Chunks + Pair Input + Joint Heads)...")
    full_stats = benchmark_full_pipeline(step6_model, tokenizer, bench_doc, device=device, n_iters=30)
    print(f"  Full Pipeline Latency: {full_stats['mean_latency_ms']:.2f} ms ({full_stats['num_chunks_processed']} chunks)")
    print(f"  Throughput: {full_stats['throughput_docs_sec']:.2f} docs/s (p95: {full_stats['p95_latency_ms']:.2f} ms)")

    # Construct Table 3.8 Artifact
    table_3_8 = {
        "device": device.type,
        "document_tokens_approx": 2000,
        "baseline": baseline_stats,
        "full_pipeline": full_stats,
        "latency_increase_factor": full_stats["mean_latency_ms"] / max(1e-5, baseline_stats["mean_latency_ms"]),
    }

    out_file = Path("results/table_3_8_latency.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(table_3_8, f, indent=2)

    print(f"\n=======================================================")
    print(f"  [Table 3.8] Detector Inference Latency (~2k doc)")
    print(f"=======================================================")
    print(f"  Baseline (Single 512):      {baseline_stats['mean_latency_ms']:8.2f} ms | {baseline_stats['throughput_docs_sec']:6.2f} docs/s")
    print(f"  Full Pipeline (Step 6):     {full_stats['mean_latency_ms']:8.2f} ms | {full_stats['throughput_docs_sec']:6.2f} docs/s ({full_stats['num_chunks_processed']} chunks)")
    print(f"  Latency Scaling Factor:     {table_3_8['latency_increase_factor']:8.2f}x")
    print(f"=======================================================")
    print(f"Latency profile saved to: {out_file}")


if __name__ == "__main__":
    main()
