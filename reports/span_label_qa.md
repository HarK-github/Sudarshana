# Span-label QA and validation split

Redaction threshold, `lambda_tok`, and false-positive calibration read `data/processed/val.jsonl`. The prototype file `data/splits/val.jsonl` is the set that produced redaction threshold 0.35.

Hard failures: 0. Warnings: 1.

## Counts

| Split | Rows | Positives | Negatives |
|---|---:|---:|---:|
| train | 244107 | 178262 | 65845 |
| val | 60639 | 44406 | 16233 |
| test | 62859 | 41960 | 20899 |

### By source

- **train**: agentdojo 1278, bipia 240800, hard_negative 50, tensortrust 1979
- **val**: agentdojo 83, bipia 60200, hard_negative 8, tensortrust 348
- **test**: agentdojo 300, bipia 60200, hard_negative 800, tensortrust 1559

## Validation attack types

- **agentdojo**: injection_task_0 (40)
- **bipia**: Cryptocurrency Mining (750), Device and Driver Enumeration (750), Malware Distribution (14250), Programming Help (14250), Space Removal & Grouping (14250)
- **tensortrust**: prompt_hijack (116)

Validation BIPIA names: Cryptocurrency Mining, Device and Driver Enumeration, Malware Distribution, Programming Help, Space Removal & Grouping.
Positive `attack_type_id` values are pairwise disjoint across train, val, and test.

## Spans on validation

- Positive rows: 44406
- Spans: 44412
- Multi-span rows: 6 (0.0001)
- Span length mean / median / min / max: 74.4 / 63.0 / 18 / 2477
- Position: {'end': 14750, 'start': 14750, 'middle': 14750, 'tool_output': 40, 'user_message': 116}
- Modality: {'text': 42906, 'code': 1500}

## Token coverage on validation

Coverage is the share of span characters that sit inside a supervised `B-INJ` or `I-INJ` token, macro-averaged over positive validation rows. Continuation pieces are unsupervised, so the score stays below 1 when a word is split.

- All: 0.8334 over 44406 rows
- Text: 0.8466 over 42906 rows
- Code: 0.4567 over 1500 rows
- Aligned rows: val 60639, train sample 500

## Counterfactuals excluded from token loss

- Train: 64262
- Val: 15906
- Test: 19460

These rows keep the parent document with label 0 and a null span, because the user task is the attack text. Detection training keeps them. Token supervision drops them.

## BIPIA document groups in both train and val

1000 groups. The clean copy of every 5th train-file context is validation, while injections of non-validation attack types on that context stay in train. Attack types stay disjoint.

## Warnings

- `bipia_context_overlap`: 1000 BIPIA document groups are in both train and val (bipia:code:train:0, bipia:code:train:1, bipia:code:train:10, bipia:code:train:11, bipia:code:train:12, bipia:code:train:13, bipia:code:train:14, bipia:code:train:15, bipia:code:train:16, bipia:code:train:17, bipia:code:train:18, bipia:code:train:19, bipia:code:train:2, bipia:code:train:20, bipia:code:train:21, bipia:code:train:22, bipia:code:train:23, bipia:code:train:24, bipia:code:train:25, bipia:code:train:26)
