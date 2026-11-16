"""Surface obfuscations for the held-out set and for training augmentation.

Training augmentation may call ``apply`` only after ``assert_train_family``.
``homoglyph`` is reserved for the obfuscated held-out set and must not appear
in a training batch. The held-out builder is the caller that is allowed to
use every family, and it applies them only to test rows.

Transforms are pure functions of the payload text. They do not sample.
"""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any

TRAIN_FAMILIES = ("base64", "hex", "leetspeak", "zero_width")
RESERVED_FAMILIES = ("homoglyph",)
EVAL_FAMILIES = TRAIN_FAMILIES + RESERVED_FAMILIES
CLEAN_FAMILY = "none"

_LEET = {
    "a": "4",
    "A": "4",
    "e": "3",
    "E": "3",
    "i": "1",
    "I": "1",
    "o": "0",
    "O": "0",
    "s": "5",
    "S": "5",
    "t": "7",
    "T": "7",
}

# Fixed Latin -> Cyrillic/Greek lookalikes. Characters without a lookalike stay.
_HOMOGLYPH = {
    "a": "\u0430",
    "c": "\u0441",
    "e": "\u0435",
    "o": "\u043e",
    "p": "\u0440",
    "x": "\u0445",
    "y": "\u0443",
    "i": "\u0456",
    "j": "\u0458",
    "s": "\u0455",
    "A": "\u0410",
    "B": "\u0412",
    "C": "\u0421",
    "E": "\u0415",
    "H": "\u041d",
    "I": "\u0406",
    "K": "\u041a",
    "M": "\u041c",
    "O": "\u041e",
    "P": "\u0420",
    "T": "\u0422",
    "X": "\u0425",
    "Y": "\u0423",
}

_ZERO_WIDTH = "\u200b"


def canonical_sha256(row: dict[str, Any]) -> str:
    """Hash a JSON object so a child can point back at one parent row."""
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assert_train_family(family: str) -> str:
    """Reject families that training augmentation is not allowed to use."""
    if family in RESERVED_FAMILIES:
        raise ValueError(
            f"obfuscation family {family!r} is reserved for the held-out set "
            "and cannot be used in training augmentation"
        )
    if family not in TRAIN_FAMILIES:
        raise ValueError(f"unknown training obfuscation family {family!r}")
    return family


def _base64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def _hex(text: str) -> str:
    return text.encode("utf-8").hex()


def _leetspeak(text: str) -> str:
    return "".join(_LEET.get(char, char) for char in text)


def _zero_width(text: str) -> str:
    return "".join(char + _ZERO_WIDTH for char in text)


def _homoglyph(text: str) -> str:
    return "".join(_HOMOGLYPH.get(char, char) for char in text)


_TRANSFORMS = {
    "base64": _base64,
    "hex": _hex,
    "leetspeak": _leetspeak,
    "zero_width": _zero_width,
    "homoglyph": _homoglyph,
}


def apply(text: str, family: str) -> str:
    """Encode one payload string with a named family."""
    try:
        transform = _TRANSFORMS[family]
    except KeyError as exc:
        raise ValueError(f"unknown obfuscation family {family!r}") from exc
    return transform(text)


def span_pairs(raw: Any) -> list[tuple[int, int]]:
    """Read ``[start, end]`` pairs or ``{start, end}`` dicts."""
    pairs: list[tuple[int, int]] = []
    for item in raw or []:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            pairs.append((int(item[0]), int(item[1])))
        elif isinstance(item, dict) and "start" in item and "end" in item:
            pairs.append((int(item["start"]), int(item["end"])))
        else:
            raise TypeError(f"injection span {item!r} is not a start/end pair")
    return pairs


def span_style(raw: Any) -> str:
    """Return ``dict`` when the parent stored spans as objects, else ``pair``."""
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        return "dict"
    return "pair"


def spans_as(pairs: list[tuple[int, int]], style: str) -> list[Any]:
    """Write spans in the parent's shape."""
    if style == "dict":
        return [{"start": start, "end": end} for start, end in pairs]
    return [[start, end] for start, end in pairs]


def context_text(document: str, spans: list[tuple[int, int]]) -> str:
    """Document text outside the spans, in order."""
    parts: list[str] = []
    cursor = 0
    for start, end in spans:
        parts.append(document[cursor:start])
        cursor = end
    parts.append(document[cursor:])
    return "".join(parts)


def rewrite_document(
    document: str,
    spans: list[tuple[int, int]],
    family: str,
) -> tuple[str, list[tuple[int, int]]]:
    """Replace each gold span and shift later offsets by the length change.

    Context outside the spans is copied unchanged. ``family`` ``none`` returns
    the document and spans as given.
    """
    if family == CLEAN_FAMILY:
        return document, list(spans)

    ordered = sorted(spans)
    parts: list[str] = []
    new_spans: list[tuple[int, int]] = []
    cursor = 0
    out_len = 0
    for index, (start, end) in enumerate(ordered):
        if not (0 <= start < end <= len(document)):
            raise ValueError(f"span {(start, end)} is outside a document of length {len(document)}")
        if start < cursor:
            raise ValueError(f"spans overlap or are unsorted at {(start, end)}")
        context = document[cursor:start]
        payload = document[start:end]
        if not payload.strip():
            raise ValueError(f"empty injection span {(start, end)}")
        transformed = apply(payload, family)
        if transformed == "":
            raise ValueError(f"family {family!r} emptied span {index}")
        parts.append(context)
        out_len += len(context)
        new_start = out_len
        parts.append(transformed)
        out_len += len(transformed)
        new_spans.append((new_start, out_len))
        cursor = end
    parts.append(document[cursor:])
    rewritten = "".join(parts)
    if context_text(rewritten, new_spans) != context_text(document, ordered):
        raise RuntimeError("span rewrite changed text outside the payload")
    return rewritten, new_spans
