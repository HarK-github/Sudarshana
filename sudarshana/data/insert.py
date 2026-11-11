"""BIPIA's insertion functions, plus the character span of the inserted attack.

The join and the middle-sentence draw follow ``bipia/data/utils.py`` (seeded
``random.Random`` per call, sentence span sampled from the tokenizer). When
NLTK's Punkt model is installed, middle insertion uses it. Otherwise a
punctuation splitter is used, and the span is still the text this function
inserted.
"""

from __future__ import annotations

import random
import re

_SENTENCE_BOUNDARY = re.compile(r"[.!?]+[\"']?\s+")
_PUNKT_WARNING = False


def sentence_spans(text: str) -> list[tuple[int, int]]:
    global _PUNKT_WARNING
    if text == "":
        return []
    try:
        from nltk.tokenize.punkt import PunktSentenceTokenizer

        return list(PunktSentenceTokenizer().span_tokenize(text))
    except Exception:
        if not _PUNKT_WARNING:
            _PUNKT_WARNING = True
            print(
                "NLTK Punkt is not available; middle insertion uses a punctuation splitter. "
                "Install nltk and the punkt model to match BIPIA's sentence indexes."
            )
        spans: list[tuple[int, int]] = []
        start = 0
        for match in _SENTENCE_BOUNDARY.finditer(text):
            spans.append((start, match.end()))
            start = match.end()
        if start < len(text):
            spans.append((start, len(text)))
        return spans


def _joined_span(prefix: str, attack: str, suffix: str) -> tuple[str, dict[str, int]]:
    document = "\n".join([prefix, attack, suffix]) if suffix != "" or prefix != "" else attack
    # "\n".join always places a newline before `attack` when prefix is present,
    # including when prefix is the empty string (BIPIA's insert_start / insert_middle).
    if prefix == "" and suffix == "" and document == attack:
        return document, {"start": 0, "end": len(attack)}
    start = len(prefix) + 1
    return document, {"start": start, "end": start + len(attack)}


def insert_end(context: str, attack: str, random_state: int | None = None) -> tuple[str, dict[str, int]]:
    del random_state
    document = "\n".join([context, attack])
    start = len(context) + 1
    return document, {"start": start, "end": start + len(attack)}


def insert_start(context: str, attack: str, random_state: int | None = None) -> tuple[str, dict[str, int]]:
    del random_state
    document = "\n".join([attack, context])
    return document, {"start": 0, "end": len(attack)}


def insert_middle(context: str, attack: str, random_state: int | None = None) -> tuple[str, dict[str, int]]:
    spans = sentence_spans(context)
    if not spans:
        start = len(context) // 2
    else:
        rng = random.Random(random_state)
        start, _ = rng.sample(spans, k=1)[0]
    document, span = _joined_span(context[:start], attack, context[start:])
    return document, span


INSERTIONS = {
    "end": insert_end,
    "start": insert_start,
    "middle": insert_middle,
}


def insert_attack(
    context: str,
    attack: str,
    position: str,
    random_state: int | None = None,
) -> tuple[str, dict[str, int]]:
    try:
        insert = INSERTIONS[position]
    except KeyError as exc:
        raise ValueError(f"unknown insertion position {position!r}") from exc
    document, span = insert(context, attack, random_state=random_state)
    if document[span["start"] : span["end"]] != attack:
        raise RuntimeError(f"insertion span does not match the attack at position {position}")
    return document, span
