"""Obfuscation augmentations and robust synthetic evasion generators (Task 5.1).

Conforms to Section 3.1, Section 7 and Hypothesis H2:
Generates synthetic obfuscations (leetspeak, homoglyphs, zero-width characters,
Base64 wrappers, markdown embeds) strictly on training data with 100% accurate
character span tracking to ensure clean supervision for L_tok.
"""

from __future__ import annotations

import base64
import copy
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .dataset import PromptInjectionRecord

# Character mapping dictionaries
LEET_DICT: Dict[str, str] = {
    "a": "4", "A": "4",
    "e": "3", "E": "3",
    "i": "1", "I": "1",
    "o": "0", "O": "0",
    "s": "5", "S": "5",
    "t": "7", "T": "7",
    "b": "8", "B": "8",
    "g": "9", "G": "9",
}

HOMOGLYPH_DICT: Dict[str, str] = {
    "a": "\u0430",  # Cyrillic small letter a
    "e": "\u0435",  # Cyrillic small letter ie
    "o": "\u043e",  # Cyrillic small letter o
    "p": "\u0440",  # Cyrillic small letter er
    "c": "\u0441",  # Cyrillic small letter es
    "x": "\u0445",  # Cyrillic small letter ha
    "y": "\u0443",  # Cyrillic small letter u
    "i": "\u0456",  # Cyrillic small letter byelorussian-ukrainian i
    "j": "\u0458",  # Cyrillic small letter je
    "s": "\u0455",  # Cyrillic small letter dze
    "A": "\u0410",  # Cyrillic capital letter A
    "E": "\u0415",  # Cyrillic capital letter Ie
    "O": "\u041e",  # Cyrillic capital letter O
    "P": "\u0420",  # Cyrillic capital letter Er
    "C": "\u0421",  # Cyrillic capital letter Es
    "T": "\u0422",  # Cyrillic capital letter Te
    "H": "\u041d",  # Cyrillic capital letter En
    "M": "\u041c",  # Cyrillic capital letter Em
    "B": "\u0412",  # Cyrillic capital letter Ve
    "K": "\u041a",  # Cyrillic capital letter Ka
}

ZERO_WIDTH_CHARS = ["\u200b", "\u200c", "\u200d", "\ufeff"]


def apply_leetspeak(
    text: str,
    spans: List[Tuple[int, int]],
    rate: float = 0.6,
    rng: Optional[random.Random] = None,
) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Substitutes characters with leetspeak equivalents.
    Preserves 1:1 character lengths, so span offsets remain strictly identical.
    """
    r = rng or random.Random()
    char_list = list(text)

    # If spans exist, prioritize perturbing inside injection spans
    target_indices = set()
    if spans:
        for s, e in spans:
            for idx in range(s, min(e, len(text))):
                target_indices.add(idx)
    else:
        target_indices = set(range(len(text)))

    for idx in target_indices:
        c = char_list[idx]
        if c in LEET_DICT and r.random() < rate:
            char_list[idx] = LEET_DICT[c]

    return "".join(char_list), list(spans)


def apply_homoglyphs(
    text: str,
    spans: List[Tuple[int, int]],
    rate: float = 0.6,
    rng: Optional[random.Random] = None,
) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Substitutes ASCII Latin characters with visually indistinguishable Cyrillic lookalikes.
    Preserves 1:1 character length, so span offsets remain strictly identical.
    """
    r = rng or random.Random()
    char_list = list(text)

    target_indices = set()
    if spans:
        for s, e in spans:
            for idx in range(s, min(e, len(text))):
                target_indices.add(idx)
    else:
        target_indices = set(range(len(text)))

    for idx in target_indices:
        c = char_list[idx]
        if c in HOMOGLYPH_DICT and r.random() < rate:
            char_list[idx] = HOMOGLYPH_DICT[c]

    return "".join(char_list), list(spans)


def apply_zero_width(
    text: str,
    spans: List[Tuple[int, int]],
    rate: float = 0.35,
    rng: Optional[random.Random] = None,
) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Inserts zero-width invisible characters (\u200b, \u200c, etc.) into text.
    Maintains exact character offset remapping to track injection spans.
    """
    r = rng or random.Random()
    new_chars: List[str] = []
    char_map: List[int] = []  # orig_idx -> new_idx

    target_indices = set()
    if spans:
        for s, e in spans:
            for idx in range(s, min(e, len(text))):
                target_indices.add(idx)
    else:
        target_indices = set(range(len(text)))

    for idx, c in enumerate(text):
        char_map.append(len(new_chars))
        new_chars.append(c)
        if idx in target_indices and c.isalnum() and r.random() < rate:
            zw = r.choice(ZERO_WIDTH_CHARS)
            new_chars.append(zw)

    char_map.append(len(new_chars))  # for end boundary

    # Remap spans
    new_spans: List[Tuple[int, int]] = []
    for s, e in spans:
        s_clamped = min(max(0, s), len(text))
        e_clamped = min(max(0, e), len(text))
        new_s = char_map[s_clamped]
        new_e = char_map[e_clamped]
        new_spans.append((new_s, new_e))

    return "".join(new_chars), new_spans


def apply_base64_wrapper(
    text: str,
    spans: List[Tuple[int, int]],
    rng: Optional[random.Random] = None,
) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Encodes the injection payload in Base64 and embeds it with standard wrapper syntax.
    If no injection span exists (benign text), encodes a random sentence.
    """
    if not spans:
        # Benign: encode a harmless sentence in middle
        words = text.split()
        if len(words) < 8:
            return text, []
        mid = len(words) // 2
        benign_chunk = " ".join(words[mid : mid + 4])
        b64_str = base64.b64encode(benign_chunk.encode("utf-8")).decode("utf-8")
        wrapped = f"BASE64: {b64_str}"
        new_text = text.replace(benign_chunk, wrapped, 1)
        return new_text, []

    # For positive injected sample, wrap the first injection span
    s, e = spans[0]
    payload = text[s:e]
    b64_str = base64.b64encode(payload.encode("utf-8")).decode("utf-8")
    wrapped = f"BASE64: {b64_str}"

    new_text = text[:s] + wrapped + text[e:]
    new_spans = [(s, s + len(wrapped))]
    return new_text, new_spans


def apply_markdown_embed(
    text: str,
    spans: List[Tuple[int, int]],
    rng: Optional[random.Random] = None,
) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Wraps payload inside markdown comment or hidden title markup.
    """
    r = rng or random.Random()
    style = r.choice(["comment", "link_title", "code_block"])

    if not spans:
        # Benign comment
        comment = "<!-- Note: Documentation reviewed and verified. -->\n"
        return comment + text, []

    s, e = spans[0]
    payload = text[s:e]

    if style == "comment":
        wrapped = f"<!-- SYSTEM_NOTE: {payload} -->"
    elif style == "link_title":
        wrapped = f'[System Guidelines](http://internal.org/security "{payload}")'
    else:
        wrapped = f"```system-meta\n{payload}\n```"

    new_text = text[:s] + wrapped + text[e:]
    new_spans = [(s, s + len(wrapped))]
    return new_text, new_spans


def augment_record(
    record: PromptInjectionRecord,
    obf_type: str,
    rng: Optional[random.Random] = None,
    suffix: str = "aug",
) -> PromptInjectionRecord:
    """
    Transforms a single PromptInjectionRecord with the specified obfuscation type.
    Accurately preserves all metadata and remapped character spans.
    """
    text = record.text
    spans = record.injection_char_spans

    if obf_type == "leetspeak":
        aug_text, aug_spans = apply_leetspeak(text, spans, rng=rng)
    elif obf_type == "homoglyph":
        aug_text, aug_spans = apply_homoglyphs(text, spans, rng=rng)
    elif obf_type == "zero_width":
        aug_text, aug_spans = apply_zero_width(text, spans, rng=rng)
    elif obf_type == "base64":
        aug_text, aug_spans = apply_base64_wrapper(text, spans, rng=rng)
    elif obf_type == "markdown_embed":
        aug_text, aug_spans = apply_markdown_embed(text, spans, rng=rng)
    else:
        aug_text, aug_spans = text, spans

    return PromptInjectionRecord(
        sample_id=f"{record.sample_id}_{suffix}_{obf_type}",
        doc_id=f"{record.doc_id}_{suffix}_{obf_type}",
        user_task=record.user_task,
        text=aug_text,
        label=record.label,
        injection_char_spans=aug_spans,
        attack_type_id=record.attack_type_id,
        attack_category=record.attack_category,
        position_bucket=record.position_bucket,
        domain=record.domain,
        raw_metadata={
            **record.raw_metadata,
            "obfuscation_type": obf_type,
            "is_augmented": True,
        },
    )


def generate_obfuscated_training_set(
    records: Sequence[PromptInjectionRecord],
    augment_ratio: float = 0.5,
    seed: int = 42,
) -> List[PromptInjectionRecord]:
    """
    Generates synthetic obfuscations for training split (Ablation Step 6).
    Applies diverse obfuscations to both injected and benign samples.
    """
    rng = random.Random(seed)
    obf_types = ["leetspeak", "homoglyph", "zero_width", "base64", "markdown_embed"]

    augmented_records: List[PromptInjectionRecord] = list(records)

    for idx, rec in enumerate(records):
        if rng.random() < augment_ratio:
            chosen_obf = rng.choice(obf_types)
            aug_rec = augment_record(rec, chosen_obf, rng=rng, suffix=f"obf{idx}")
            augmented_records.append(aug_rec)

    return augmented_records


def generate_obfuscation_eval_splits(
    records: Sequence[PromptInjectionRecord],
    seed: int = 42,
) -> Dict[str, List[PromptInjectionRecord]]:
    """
    Creates test sets perturbed under each distinct obfuscation technique
    to compute Table 3.5 (Obfuscation Robustness).
    """
    obf_types = ["leetspeak", "homoglyph", "zero_width", "base64", "markdown_embed"]
    splits: Dict[str, List[PromptInjectionRecord]] = {"clean": list(records)}

    for obf in obf_types:
        rng = random.Random(seed)
        obf_list: List[PromptInjectionRecord] = []
        for rec in records:
            obf_rec = augment_record(rec, obf, rng=rng, suffix=f"test_{obf}")
            obf_list.append(obf_rec)
        splits[obf] = obf_list

    return splits
