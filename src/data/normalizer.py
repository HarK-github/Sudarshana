"""Text normalization pipeline (Hypothesis H2) conforming to Section 3.1 & Section 7.

Normalizes raw text to defeat character-level evasion techniques:
1. Unicode NFKC normalization (replaces homoglyphs, full-width characters, ligatures).
2. Zero-width character stripping (removes invisible whitespace, joiners, bi-di overrides).
3. Base64 / Hex payload decoding (decodes obfuscated instruction strings to cleartext).
4. Offset mapping: re-anchors gold character spans from raw coordinates to normalized coordinates.
"""

from __future__ import annotations

import base64
import binascii
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple


# Regex for zero-width and invisible control characters
ZERO_WIDTH_RE = re.compile(
    r"[\u200B-\u200D\uFEFF\u200E\u200F\u202A-\u202E\u2060-\u2064\u00AD]"
)

# Regex to detect candidate Base64 strings (at least 16 valid characters with valid padding)
B64_CANDIDATE_RE = re.compile(
    r"(?:(?:base64|b64)\s*:\s*)?([A-Za-z0-9+/]{16,}={0,2})",
    re.IGNORECASE,
)

# Regex to detect candidate Hex byte strings (at least 8 pairs of hex digits)
HEX_CANDIDATE_RE = re.compile(
    r"(?:(?:hex)\s*:\s*)?((?:[0-9a-fA-F]{2}[\s,:]*){8,})",
    re.IGNORECASE,
)


def is_meaningful_decoded_text(s: str) -> bool:
    """Checks whether decoded bytes represent valid printable human-readable text."""
    if not s or len(s) < 4:
        return False
    # Check printable ratio
    printable_chars = sum(1 for c in s if c.isprintable() or c in "\r\n\t")
    if printable_chars / len(s) < 0.85:
        return False
    # Check that it contains at least one space or standard word characters
    has_letters = any(c.isalpha() for c in s)
    return has_letters


def decode_base64_string(b64_str: str) -> Optional[str]:
    """Safely attempts to decode a candidate Base64 string to printable UTF-8."""
    # Ensure correct padding length
    padded = b64_str + "=" * ((4 - len(b64_str) % 4) % 4)
    try:
        raw_bytes = base64.b64decode(padded, validate=True)
        decoded = raw_bytes.decode("utf-8", errors="strict")
        if is_meaningful_decoded_text(decoded):
            return decoded
    except Exception:
        pass
    return None


def decode_hex_string(hex_str: str) -> Optional[str]:
    """Safely attempts to decode a candidate hex string to printable UTF-8."""
    cleaned = re.sub(r"[\s,:]+", "", hex_str)
    if len(cleaned) % 2 != 0:
        return None
    try:
        raw_bytes = binascii.unhexlify(cleaned)
        decoded = raw_bytes.decode("utf-8", errors="strict")
        if is_meaningful_decoded_text(decoded):
            return decoded
    except Exception:
        pass
    return None


class TextNormalizer:
    """
    Normalizes input text and maintains character offset maps to accurately re-anchor
    injection character spans for token loss supervision (L_tok).
    """

    def __init__(
        self,
        nfkc: bool = True,
        strip_zero_width: bool = True,
        decode_b64: bool = True,
        decode_hex: bool = True,
    ):
        self.nfkc = nfkc
        self.strip_zero_width = strip_zero_width
        self.decode_b64 = decode_b64
        self.decode_hex = decode_hex

    def normalize(
        self,
        text: str,
        spans: Optional[List[Tuple[int, int]]] = None,
    ) -> Tuple[str, List[Tuple[int, int]]]:
        """
        Normalizes text while tracking span coordinate transforms.

        Args:
            text: Original document or chunk string.
            spans: List of (start_char, end_char) injection spans in original text.

        Returns:
            Tuple of (normalized_text, re_anchored_spans).
        """
        if not text:
            return "", []

        spans_in = list(spans or [])

        # Step 1: Decode Base64 & Hex if enabled
        cur_text, cur_spans = text, spans_in
        if self.decode_b64:
            cur_text, cur_spans = self._decode_encoded_patterns(
                cur_text, cur_spans, B64_CANDIDATE_RE, decode_base64_string
            )
        if self.decode_hex:
            cur_text, cur_spans = self._decode_encoded_patterns(
                cur_text, cur_spans, HEX_CANDIDATE_RE, decode_hex_string
            )

        # Step 2: Strip Zero-Width Characters and apply NFKC
        norm_chars: List[str] = []
        # orig_to_norm_map maps each original char index in cur_text to the index in norm_text
        orig_to_norm: List[int] = [0] * (len(cur_text) + 1)

        norm_pos = 0
        for i, char in enumerate(cur_text):
            orig_to_norm[i] = norm_pos

            # Check zero-width filter
            if self.strip_zero_width and ZERO_WIDTH_RE.match(char):
                continue

            # Check NFKC normalization
            if self.nfkc:
                normed = unicodedata.normalize("NFKC", char)
            else:
                normed = char

            norm_chars.append(normed)
            norm_pos += len(normed)

        orig_to_norm[len(cur_text)] = norm_pos
        norm_text = "".join(norm_chars)

        # Remap spans
        final_spans: List[Tuple[int, int]] = []
        for s_start, s_end in cur_spans:
            s_start_clamped = max(0, min(s_start, len(cur_text)))
            s_end_clamped = max(0, min(s_end, len(cur_text)))

            new_start = orig_to_norm[s_start_clamped]
            new_end = orig_to_norm[s_end_clamped]

            if new_end > new_start:
                final_spans.append((new_start, new_end))

        return norm_text, final_spans

    def _decode_encoded_patterns(
        self,
        text: str,
        spans: List[Tuple[int, int]],
        pattern: re.Pattern,
        decode_fn: Any,
    ) -> Tuple[str, List[Tuple[int, int]]]:
        """Finds candidate encoded substrings, decodes them if valid, and updates text and spans."""
        matches = list(pattern.finditer(text))
        if not matches:
            return text, spans

        out_parts: List[str] = []
        orig_to_new = [0] * (len(text) + 1)
        last_idx = 0
        current_new_idx = 0

        for match in matches:
            match_start, match_end = match.span()
            matched_str = match.group(1) if match.lastindex and match.lastindex >= 1 else match.group(0)

            decoded = decode_fn(matched_str)
            if decoded is not None:
                # Copy prefix before match
                prefix = text[last_idx:match_start]
                out_parts.append(prefix)
                for i in range(last_idx, match_start):
                    orig_to_new[i] = current_new_idx + (i - last_idx)
                current_new_idx += len(prefix)

                # Insert decoded replacement
                replacement = decoded
                out_parts.append(replacement)
                for i in range(match_start, match_end):
                    orig_to_new[i] = current_new_idx
                current_new_idx += len(replacement)

                last_idx = match_end

        # Copy remainder of text
        suffix = text[last_idx:]
        out_parts.append(suffix)
        for i in range(last_idx, len(text) + 1):
            orig_to_new[i] = current_new_idx + (i - last_idx)

        new_text = "".join(out_parts)

        # Remap spans
        remapped_spans: List[Tuple[int, int]] = []
        for s_start, s_end in spans:
            s_start_clamped = max(0, min(s_start, len(text)))
            s_end_clamped = max(0, min(s_end, len(text)))

            new_start = orig_to_new[s_start_clamped]
            new_end = orig_to_new[s_end_clamped]
            if new_end > new_start:
                remapped_spans.append((new_start, new_end))

        return new_text, remapped_spans


def normalize_text(text: str) -> str:
    """Convenience function for normalizing text without span tracking."""
    normalizer = TextNormalizer()
    norm_text, _ = normalizer.normalize(text)
    return norm_text
