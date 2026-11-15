"""Latency benchmarking for P5.

Builds a ~2000-token document from concatenated test docs, times each
available pipeline stage over 20 runs (3 warmups) and reports p50/p95 in ms.
Token count estimated as words * 1.3 (project tokenizer used when available).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from .p5_common import load_split, CHECKPOINTS_DIR, REPO_ROOT


# ---------------------------------------------------------------------------
# Tokenizer helper
# ---------------------------------------------------------------------------

def _get_tokenizer() -> Any | None:
    """Return project tokenizer if a checkpoint dir with tokenizer.json exists."""
    for ckpt_dir in CHECKPOINTS_DIR.iterdir():
        if not ckpt_dir.is_dir():
            continue
        if (ckpt_dir / "tokenizer.json").exists():
            try:
                from transformers import AutoTokenizer
                return AutoTokenizer.from_pretrained(str(ckpt_dir))
            except Exception:
                return None
    return None


def _word_estimate(text: str) -> int:
    """Estimate token count as words * 1.3."""
    return int(len(text.split()) * 1.3)


def build_benchmark_doc(target_tokens: int = 2000) -> str:
    """Concatenate test documents until ~target_tokens reached."""
    tokenizer = _get_tokenizer()
    records = load_split("test")
    chunks: list[str] = []
    total = 0
    for rec in records:
        doc = rec["document"]
        if tokenizer is not None:
            try:
                n = len(tokenizer.encode(doc, add_special_tokens=False))
            except Exception:
                n = _word_estimate(doc)
        else:
            n = _word_estimate(doc)
        chunks.append(doc)
        total += n
        if total >= target_tokens:
            break
    return "\n\n".join(chunks)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _time_fn(fn: Any, *args: Any, n_runs: int = 20, n_warmup: int = 3) -> dict[str, float]:
    """Time fn(*args) over n_runs calls, report p50/p95 in ms."""
    for _ in range(n_warmup):
        fn(*args)
    timings: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(*args)
        timings.append((time.perf_counter() - t0) * 1000)
    arr = np.array(timings)
    return {
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "mean_ms": round(float(np.mean(arr)), 2),
        "n_runs": n_runs,
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_latency_benchmark() -> dict[str, Any]:
    """Benchmark all available pipeline stages on a ~2000-token document.

    Returns dict keyed by stage name with p50/p95 timing or N/A with reason.
    """
    doc = build_benchmark_doc(target_tokens=2000)
    actual_tokens = _word_estimate(doc)
    results: dict[str, Any] = {
        "benchmark_doc_word_estimate": actual_tokens,
        "benchmark_doc_chars": len(doc),
        "stages": {},
    }

    # Stage 1: normalize (pysbd sentence segmentation only)
    try:
        import pysbd
        seg = pysbd.Segmenter(language="en", clean=False)
        results["stages"]["normalize_pysbd"] = _time_fn(seg.segment, doc)
    except ImportError:
        results["stages"]["normalize_pysbd"] = {"na_reason": "pysbd not available"}

    # Stage 2: redact (with gold oracle spans — empty spans = no-op; use a fixed small span)
    try:
        from sudarshana.redaction.redact import redact
        # Use a fixed 50-char span as a representative redaction target
        test_span: list[tuple[int, int]] = [(0, min(50, len(doc)))]
        results["stages"]["redact"] = _time_fn(redact, doc, test_span)
    except Exception as exc:
        results["stages"]["redact"] = {"na_reason": f"redact unavailable: {exc}"}

    # Stage 3: tokenizer encode (project tokenizer if available)
    tokenizer = _get_tokenizer()
    if tokenizer is not None:
        def _tok_encode(text: str) -> Any:
            return tokenizer(text, truncation=True, max_length=512, return_tensors="pt")

        results["stages"]["tokenizer_encode"] = _time_fn(_tok_encode, doc)
    else:
        results["stages"]["tokenizer_encode"] = {"na_reason": "no tokenizer found"}

    # Stage 4: model forward (only if weights exist)
    import os
    has_weights = False
    weight_exts = {".bin", ".safetensors", ".pt"}
    for ckpt_dir in CHECKPOINTS_DIR.iterdir():
        if not ckpt_dir.is_dir():
            continue
        for f in ckpt_dir.iterdir():
            if f.suffix in weight_exts:
                has_weights = True
                break

    if has_weights and tokenizer is not None:
        try:
            from sudarshana.redaction.redact import load_prototype_model, predict_spans
            model, tok = load_prototype_model()
            if model is not None:
                inputs = tokenizer(doc, truncation=True, max_length=512, return_tensors="pt")
                import torch

                def _forward() -> Any:
                    with torch.no_grad():
                        return model(**inputs)

                results["stages"]["model_forward"] = _time_fn(_forward)
            else:
                results["stages"]["model_forward"] = {"na_reason": "model load returned None"}
        except Exception as exc:
            results["stages"]["model_forward"] = {"na_reason": f"model load failed: {exc}"}
    else:
        results["stages"]["model_forward"] = {
            "na_reason": "no weights — checkpoints/ contain only config+tokenizer files"
        }

    return results
