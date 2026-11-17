# Obfuscated held-out card

Surface-obfuscated copy of the test split. Evaluation reads this file.
Training must not. Reserved families must not be used as training augmentation.

- Source: `data/splits/test.jsonl`
- Source sha256: `e6055150334c26f5c46946526b16885f07ab681dc54651fbe725692c220c1b0d`
- Seed: 20261115 (recorded; the current transforms do not sample)
- Git revision: `77bed2f828b43a5e61c09fbb75a16f49df99af3e`
- Rows: 706
- Injected parents: 125
- Clean controls: 81
- Reserved families: `homoglyph`
- Clean controls whose document text also appears in train: 7
- Clean controls whose document text also appears in val: 3

## Counts by family

| Family | Rows | Unchanged payloads |
|--------|------|--------------------|
| none | 81 | — |
| base64 | 125 | 0 |
| hex | 125 | 0 |
| leetspeak | 125 | 0 |
| zero_width | 125 | 0 |
| homoglyph | 125 | 0 |

Protocol: `docs/eval_protocol_obfuscated.md`
