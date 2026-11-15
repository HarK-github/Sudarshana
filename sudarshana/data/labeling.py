"""Token labeling and offset mapping alignment for Sudarshana.

Aligns character-level injection spans to token-level BIO labels (O, B-INJ, I-INJ).
Enforces rules:
- Special and padding tokens get -100.
- Subword continuation tokens get -100.
- First subword of an injection gets B-INJ (1), subsequent words get I-INJ (2).
- Non-injected tokens get O (0).
"""

from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    from transformers import PreTrainedTokenizerFast
except ImportError:
    torch = None  # type: ignore[assignment]
    PreTrainedTokenizerFast = None  # type: ignore[assignment,misc]

LABEL2ID = {"O": 0, "B-INJ": 1, "I-INJ": 2}
ID2LABEL = {0: "O", 1: "B-INJ", 2: "I-INJ"}
PAD_LABEL_ID = -100


def coerce_spans(raw: Any) -> list[tuple[int, int]]:
    """Normalize schema 1.1 dict spans and prototype ``[start, end]`` pairs."""
    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = [raw]
    spans: list[tuple[int, int]] = []
    for item in raw:
        if isinstance(item, dict):
            spans.append((int(item["start"]), int(item["end"])))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            spans.append((int(item[0]), int(item[1])))
        else:
            raise TypeError("span must be {start, end} or [start, end]")
    return spans


def build_model_input(
    document: str,
    user_task: Optional[str] = None
) -> str:
    """
    Input builder hook.
    Currently returns the document window alone.
    P3 can modify this hook to format (user_task, document) pair input without altering downstream logic.
    """
    if user_task is not None and len(user_task.strip()) > 0:
        # Placeholder format for future P3 pair input integration
        # For now, default prototype input operates on document alone as specified in brief
        return document
    return document


def _span_index_at(spans: List[Tuple[int, int]], char_start: int) -> Optional[int]:
    for index, (span_start, span_end) in enumerate(spans):
        if span_start <= char_start < span_end:
            return index
    return None


def align_spans_to_token_labels(
    offset_mapping: List[Tuple[int, int]],
    word_ids: List[Optional[int]],
    spans: List[Tuple[int, int]],
    text: Optional[str] = None,
) -> List[int]:
    """
    Convert character offsets to token-level BIO labels.

    A word-initial token is positive when its start offset lies inside a span.
    The first such word of a span is B-INJ and later words are I-INJ.
    Specials and subword continuations are -100.

    When ``text`` is provided, a word-initial token that starts on whitespace
    immediately before a span and then enters that span is B-INJ if the span
    would otherwise have no word-initial token. SentencePiece does this for a
    payload glued to the preceding newline with no spaces inside it.
    """
    labels: List[int] = []
    prev_word_id = None
    in_injection = False
    word_initial: List[int] = []

    for idx, (char_start, char_end) in enumerate(offset_mapping):
        word_id = word_ids[idx]

        # Rule 1: Special tokens (word_id is None) get -100
        if word_id is None or (char_start == 0 and char_end == 0):
            labels.append(PAD_LABEL_ID)
            continue

        # Rule 2: Subword continuation pieces get -100
        if word_id == prev_word_id:
            labels.append(PAD_LABEL_ID)
            continue

        prev_word_id = word_id
        word_initial.append(idx)

        # Token is inside a span when its start offset falls in [span_start, span_end).
        is_token_injected = False
        is_span_start = False

        for span_start, span_end in spans:
            if char_start >= span_start and char_start < span_end:
                is_token_injected = True
                if not in_injection or char_start == span_start:
                    is_span_start = True
                break

        if is_token_injected:
            if is_span_start:
                labels.append(LABEL2ID["B-INJ"])
                in_injection = True
            else:
                labels.append(LABEL2ID["I-INJ"])
        else:
            labels.append(LABEL2ID["O"])
            in_injection = False

    if text is None:
        return labels

    covered = {
        span_index
        for idx in word_initial
        if (span_index := _span_index_at(spans, offset_mapping[idx][0])) is not None
        and labels[idx] in (LABEL2ID["B-INJ"], LABEL2ID["I-INJ"])
    }
    for span_index, (span_start, span_end) in enumerate(spans):
        if span_index in covered:
            continue
        for idx in word_initial:
            char_start, char_end = offset_mapping[idx]
            if not (char_start < span_start < char_end) or labels[idx] != LABEL2ID["O"]:
                continue
            if text[char_start:span_start].strip() == "":
                labels[idx] = LABEL2ID["B-INJ"]
                break

    return labels


def tokenize_and_align_document(
    document: str,
    spans: List[Tuple[int, int]],
    tokenizer: PreTrainedTokenizerFast,
    user_task: Optional[str] = None,
    max_length: int = 256,
    stride: int = 64
) -> List[Dict[str, Any]]:
    """
    Cut document into 256-token sliding windows with 64-token overlap and align token labels.
    Handles documents of arbitrary length by sliding windows across the full token sequence.
    
    Returns a list of window dicts:
        - input_ids: List[int]
        - attention_mask: List[int]
        - labels: List[int]
        - offset_mapping: List[Tuple[int, int]]
        - window_idx: int
        - num_windows: int
    """
    text_to_tokenize = build_model_input(document, user_task)

    # Tokenize the full document once with offset mapping
    full_enc = tokenizer(
        text_to_tokenize,
        truncation=False,
        return_offsets_mapping=True,
        return_tensors=None
    )

    all_input_ids = full_enc["input_ids"]
    all_attention_mask = full_enc["attention_mask"]
    all_offsets = full_enc["offset_mapping"]
    all_word_ids = full_enc.word_ids()

    # Align labels for the full document
    all_labels = align_spans_to_token_labels(
        all_offsets, all_word_ids, spans, text=text_to_tokenize
    )

    step = max_length - stride  # 192 tokens step
    windows = []
    total_tokens = len(all_input_ids)

    # Slice into windows of max_length with stride overlap
    for start_idx in range(0, total_tokens, step):
        end_idx = min(start_idx + max_length, total_tokens)
        win_ids = all_input_ids[start_idx:end_idx]
        win_mask = all_attention_mask[start_idx:end_idx]
        win_labels = all_labels[start_idx:end_idx]
        win_offsets = all_offsets[start_idx:end_idx]

        # Pad to max_length if necessary
        pad_len = max_length - len(win_ids)
        if pad_len > 0:
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            win_ids = win_ids + [pad_id] * pad_len
            win_mask = win_mask + [0] * pad_len
            win_labels = win_labels + [PAD_LABEL_ID] * pad_len
            win_offsets = win_offsets + [(0, 0)] * pad_len

        windows.append({
            "input_ids": win_ids,
            "attention_mask": win_mask,
            "labels": win_labels,
            "offset_mapping": win_offsets,
            "window_idx": len(windows),
            "start_token_idx": start_idx,
            "end_token_idx": end_idx
        })

        if end_idx == total_tokens:
            break

    # Add num_windows to all
    for w in windows:
        w["num_windows"] = len(windows)

    return windows
