> Prototype corpus used by the token-head prototype. Tuning reads `data/processed/val.jsonl`.

# Sudarshana Unified Dataset Card

## 1. Overview and Ownership
- **Schema**: `(user_task, document, label, injection_span, attack_type)`
- **Primary Source**: Microsoft BIPIA (`microsoft/BIPIA`), MIT License
- **Hard Negatives**: Technical documentation, Linux tutorials, cooking recipes, action-oriented business emails
- **Spans**: Character offsets `(start_char, end_char)` in original text, verified with 100% round-trip match

## 2. Dataset Counts per Split
- **Train**: 386 records (300 injected, 86 benign)
- **Validation**: 98 records (75 injected, 23 benign)
- **Held-out Test**: 206 records (125 injected, 81 benign)

## 3. Attack Type Split Strategy
- **Zero-leakage split**: Test split contains exclusively held-out attack types:
  - 15 Held-out Text Attack Types (Task Automation, Business Intelligence, Substitution Ciphers, Base Encoding, etc.)
  - 10 Held-out Code Attack Types (Data Eavesdropping, Traffic Analysis, Keylogging, Screen Scraping, etc.)
- Train/Val splits draw from disjoint training attack types (Information Retrieval, Content Creation, Cookie Theft, etc.)

## 4. Positions and Insertion Styles
### Positions in Test Split:
- `none`: 81
- `start`: 42
- `middle`: 30
- `end`: 53

### Insertion Styles in Test Split:
- `none`: 81
- `wrapped_markup`: 42
- `sentence_boundary`: 42
- `inline`: 41

## 5. Tasks and Domains
### Test Domain Breakdown:
- `text`: 131
- `code`: 75

## 6. Known Limitations
- Single-turn document retrieval only; multi-turn agent traces not included.
- Code attacks evaluated on Python Stack Overflow contexts; compiled languages not covered.
- The test split itself only contains BIPIA's own wrappers and ciphers. Surface obfuscation of those payloads is the separate held-out file in section 7.

## 7. Obfuscated held-out set
Built from this test split by `python -m sudarshana.data.build_obfuscated --config configs/obfuscated_heldout.yaml`.

- File: `data/splits/obfuscated.jsonl`
- Card: `data/splits/OBFUSCATED_CARD.md`
- Protocol: `docs/eval_protocol_obfuscated.md`
- Each injected test row is rewritten once per family (`base64`, `hex`, `leetspeak`, `zero_width`, `homoglyph`). Only the gold span changes. Clean test rows are copied once as the false-positive control.
- `homoglyph` is reserved for this set. Training augmentation must not use it.
- Training must not read `obfuscated.jsonl`.
