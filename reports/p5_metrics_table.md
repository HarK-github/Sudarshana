# P5 Metrics Table

**Split**: test | **Seed**: 42

| Predictor | Recall | FPR (test) | FPR (hard-neg) | Span F1 | PSR | ROUGE-1 F1 | Latency p50 (ms) | Latency p95 (ms) |
|-----------|--------|------------|----------------|---------|-----|------------|-----------------|-----------------|
| none | 0.0000 | 0.0000 | 0.0000 | N/A | 1.0000 | 0.8962 | 0.1 | 0.1 |
| oracle | 1.0000 | 0.0000 | 0.0000 | N/A | 0.0000 | 0.8996 | 0.1 | 0.1 |
| keyword_heuristic | 0.0400 | 0.0123 | 0.0000 | N/A | 1.0000 | 0.8967 | 0.1 | 0.1 |
| token_head | N/A (no weights — checkpoints/ contain only config+tokenizer files) | N/A | N/A | N/A (no weights — checkpoints/ contain only config+tokenizer files) | N/A (no weights — checkpoints/ contain only config+tokenizer files) | N/A (no weights — checkpoints/ contain only config+tokenizer files) | N/A (no weights — checkpoints/ contain only config+tokenizer files) | N/A (no weights — checkpoints/ contain only config+tokenizer files) |