"""Obfuscation augmentation for training data — P3 deliverable.

Applies lightweight obfuscations to injection spans only (document text),
so the model learns to detect obfuscated attacks without drifting on clean text.

Augmentation strategies:
  - base64 encode words
  - homoglyph substitution (latin → lookalike unicode)
  - zero-width char insertion
  - leet-speak substitution
  - random capitalisation

Each sample is independently augmented with probability `p` per strategy.
Span offsets are recomputed after augmentation.
"""

from __future__ import annotations

import base64
import random
import re
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Homoglyph table (small but representative)
# ---------------------------------------------------------------------------
_HOMOGLYPHS: dict[str, str] = {
    "a": "а",  # Cyrillic а
    "e": "е",  # Cyrillic е
    "o": "о",  # Cyrillic о
    "p": "р",  # Cyrillic р
    "c": "с",  # Cyrillic с
    "i": "і",  # Cyrillic і
    "x": "х",  # Cyrillic х
    "A": "А",
    "E": "Е",
    "O": "О",
}

_LEET: dict[str, str] = {
    "a": "4", "e": "3", "i": "1", "o": "0",
    "s": "5", "t": "7", "g": "9", "l": "1",
}

_ZERO_WIDTH = "\u200b"   # zero-width space


# ---------------------------------------------------------------------------
# Token-level helpers
# ---------------------------------------------------------------------------

def _homoglyph(text: str, rate: float = 0.3) -> str:
    out = []
    for ch in text:
        if ch.lower() in _HOMOGLYPHS and random.random() < rate:
            out.append(_HOMOGLYPHS[ch.lower()])
        else:
            out.append(ch)
    return "".join(out)


def _leet(text: str, rate: float = 0.3) -> str:
    out = []
    for ch in text:
        if ch.lower() in _LEET and random.random() < rate:
            out.append(_LEET[ch.lower()])
        else:
            out.append(ch)
    return "".join(out)


def _zero_width_insert(text: str, rate: float = 0.15) -> str:
    out = []
    for ch in text:
        out.append(ch)
        if random.random() < rate:
            out.append(_ZERO_WIDTH)
    return "".join(out)


def _rand_caps(text: str, rate: float = 0.3) -> str:
    return "".join(
        ch.upper() if ch.isalpha() and random.random() < rate else ch
        for ch in text
    )


def _b64_words(text: str, word_rate: float = 0.4) -> str:
    """Base64-encode random whole words."""
    tokens = re.split(r"(\s+)", text)
    out = []
    for tok in tokens:
        if re.match(r"\S", tok) and random.random() < word_rate:
            encoded = base64.b64encode(tok.encode()).decode()
            out.append(encoded)
        else:
            out.append(tok)
    return "".join(out)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_STRATEGIES = [_homoglyph, _leet, _zero_width_insert, _rand_caps, _b64_words]


def augment_injection_spans(
    document: str,
    spans: List[Tuple[int, int]],
    p: float = 0.5,
    seed: Optional[int] = None,
) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Augment the injection spans in *document* and return the new document
    with updated character-level span offsets.

    Args:
        document: Original document text.
        spans:    List of (start, end) injection spans (char offsets).
        p:        Probability of applying each strategy to each span.
        seed:     Optional RNG seed for reproducibility.

    Returns:
        (augmented_document, new_spans)
    """
    if seed is not None:
        random.seed(seed)

    if not spans:
        return document, spans

    # Sort spans so we can rebuild the string left-to-right
    spans_sorted = sorted(spans)
    result_parts: List[str] = []
    new_spans: List[Tuple[int, int]] = []
    cursor = 0
    offset_shift = 0

    for s, e in spans_sorted:
        # Text before this span
        result_parts.append(document[cursor:s])

        # Apply random strategies to span text
        span_text = document[s:e]
        for strategy in _STRATEGIES:
            if random.random() < p:
                span_text = strategy(span_text)

        new_start = s + offset_shift
        new_end   = new_start + len(span_text)
        new_spans.append((new_start, new_end))
        offset_shift += len(span_text) - (e - s)

        result_parts.append(span_text)
        cursor = e

    result_parts.append(document[cursor:])
    return "".join(result_parts), new_spans


def augment_records(
    records: List[dict],
    p: float = 0.5,
    skip_negatives: bool = True,
) -> List[dict]:
    """
    Return a new list of records with augmented documents (injections only).
    Original records are NOT mutated.

    Args:
        records:        List of dataset records (schema: document, injection_span, label, ...).
        p:              Per-strategy augmentation probability.
        skip_negatives: If True, clean records (label=0) are passed through unchanged.
    """
    augmented = []
    for rec in records:
        if skip_negatives and rec.get("label", 0) == 0:
            augmented.append(rec)
            continue

        raw_spans = rec.get("injection_span") or []
        spans: List[Tuple[int, int]] = []
        for s in raw_spans:
            if isinstance(s, dict):
                spans.append((int(s["start"]), int(s["end"])))
            elif isinstance(s, (list, tuple)):
                spans.append((int(s[0]), int(s[1])))

        new_doc, new_spans = augment_injection_spans(rec["document"], spans, p=p)

        new_rec = dict(rec)
        new_rec["document"] = new_doc
        new_rec["injection_span"] = [{"start": s, "end": e} for s, e in new_spans]
        new_rec["augmented"] = True
        augmented.append(new_rec)

    return augmented
