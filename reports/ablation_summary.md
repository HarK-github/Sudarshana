# Sudarshana Ablation Ladder: Final Experimental Report

## 1. Table 3.2 (Comprehensive): Full Detection & Classification Metrics (Calibrated Operating Thresholds)
| Step | Variant | Calib $\tau^*$ | Recall | Clean FPR (N=75) | Hard-Neg FPR | HN FPR @ 90% Rec | Precision | F1 Score | Span F1 | Latency |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **step1_baseline** | 0.4973 | 60.80% | 26.67% | 0.00% | 0.00% | 0.7917 | 0.6878 | n/a | 43.56 ms |
| 2 | **step2_normalizer** | 0.5920 | 35.20% | 14.67% | 0.00% | 0.00% | 0.8000 | 0.4889 | n/a | 46.12 ms |
| 3 | **step3_chunking** | 0.6082 | 28.00% | 21.33% | 16.67% | 100.00% | 0.6731 | 0.3955 | n/a | 152.40 ms |
| 4 | **step4_pair_input** | 0.5868 | 35.20% | 22.67% | 0.00% | 100.00% | 0.7213 | 0.4731 | n/a | 158.85 ms |
| 5 | **step5_token_head** | 0.7466 | 48.80% | 6.67% | 0.00% | 83.33% | 0.9242 | 0.6387 | 34.42% | 164.20 ms |
| 6 | **step6_augmented** | 0.7162 | 61.60% | 17.33% | 0.00% | 83.33% | 0.8556 | 0.7163 | 34.42% | 166.61 ms |

## 1b. Table 3.2 (Course Specification Format — Page 9 Template)
| Step | Variant | Recall | Clean FPR (N=75) | Hard-Neg FPR @ 90% Rec | Span F1 | ASR (LLM) | ROUGE-1 | Latency |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **step1_baseline** | 60.80% | 26.67% | 0.00% | n/a | [P5 handoff] | [P5 handoff] | 43.56 ms |
| 2 | **step2_normalizer** | 35.20% | 14.67% | 0.00% | n/a | [P5 handoff] | [P5 handoff] | 46.12 ms |
| 3 | **step3_chunking** | 28.00% | 21.33% | 100.00% | n/a | [P5 handoff] | [P5 handoff] | 152.40 ms |
| 4 | **step4_pair_input** | 35.20% | 22.67% | 100.00% | n/a | [P5 handoff] | [P5 handoff] | 158.85 ms |
| 5 | **step5_token_head** | 48.80% | 6.67% | 83.33% | 34.42% | [P5 handoff] | [P5 handoff] | 164.20 ms |
| 6 | **step6_augmented** | 61.60% | 17.33% | 83.33% | 34.42% | [P5 handoff] | [P5 handoff] | 166.61 ms |

## 1c. Uncalibrated Baseline (tau = 0.50) Reference Comparison
| Step | Variant | Test Recall (0.5) | Clean FPR (0.5) | Precision (0.5) | F1 Score (0.5) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | **step1_baseline** | 54.40% | 24.00% | 0.7907 | 0.6445 |
| 2 | **step2_normalizer** | 86.40% | 57.33% | 0.7152 | 0.7826 |
| 3 | **step3_chunking** | 40.00% | 33.33% | 0.6579 | 0.4975 |
| 4 | **step4_pair_input** | 45.60% | 28.00% | 0.7308 | 0.5616 |
| 5 | **step5_token_head** | 63.20% | 33.33% | 0.7596 | 0.6900 |
| 6 | **step6_augmented** | 63.20% | 22.67% | 0.8229 | 0.7149 |

## 2. Table 3.3: Truncation vs. Chunking (Hypothesis H1)
| Variant | Start Recall | Middle Recall | End Recall |
| :--- | :---: | :---: | :---: |
| **step1_baseline** | 50.00% | 76.67% | 60.38% |
| **step2_normalizer** | 40.48% | 40.00% | 28.30% |
| **step3_chunking** | 26.19% | 26.67% | 30.19% |
| **step4_pair_input** | 35.71% | 30.00% | 37.74% |
| **step5_token_head** | 52.38% | 40.00% | 50.94% |
| **step6_augmented** | 59.52% | 66.67% | 60.38% |

## 3. Table 3.4: Recall by Attack Category (Hypothesis H3)
| Variant | Task-Irrelevant | Task-Relevant | Targeted |
| :--- | :---: | :---: | :---: |
| **step1_baseline** | 44.00% | 45.00% | 70.00% |
| **step2_normalizer** | 40.00% | 35.00% | 33.75% |
| **step3_chunking** | 32.00% | 25.00% | 27.50% |
| **step4_pair_input** | 40.00% | 30.00% | 35.00% |
| **step5_token_head** | 24.00% | 25.00% | 62.50% |
| **step6_augmented** | 40.00% | 30.00% | 76.25% |

## 4. Table 3.5: Obfuscation Robustness Suite (Hypothesis H2)
| Obfuscation Type | Baseline Recall (Step 1) | With Normalizer (Step 2) | With Normalizer & Augmentation (Step 6) |
| :--- | :---: | :---: | :---: |
| **Base64 wrapper** | 40.00% | 95.20% | 98.40% |
| **Homoglyphs** | 63.20% | 67.20% | 97.60% |
| **Zero-width characters** | 44.80% | 96.00% | 98.40% |
| **Leetspeak** | 63.20% | 96.00% | 99.20% |
| **Markdown embeds** | 58.40% | 96.00% | 98.40% |

## 5. Table 3.7: Modality Breakdown (Text QA vs. Code QA)
| Variant | Text QA Recall | Text QA FPR | Code QA Recall | Code QA FPR |
| :--- | :---: | :---: | :---: | :---: |
| **step1_baseline** | 50.67% | 23.21% | 76.00% | 28.00% |
| **step2_normalizer** | 45.33% | 16.07% | 20.00% | 8.00% |
| **step3_chunking** | 30.67% | 30.36% | 24.00% | 0.00% |
| **step4_pair_input** | 37.33% | 30.36% | 32.00% | 0.00% |
| **step5_token_head** | 25.33% | 8.93% | 84.00% | 0.00% |
| **step6_augmented** | 37.33% | 23.21% | 98.00% | 0.00% |

## 6. Table 3.8: Detector Latency Profile (~2k Document)
- **Device:** CPU
- **Baseline Latency:** 43.56 ms (22.96 docs/s)
- **Full Pipeline Latency:** 166.61 ms (6.00 docs/s, 10 chunks)
- **Scaling Factor:** 3.82x

## 7. Model Checkpoint Freeze Verification
All 6 production checkpoints frozen and preserved under `models/final_step{1..6}/`.