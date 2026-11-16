"""Inference and span redaction for Sudarshana.

Public Interface (stable):
def predict_spans(text: str, user_task: str | None = None) -> list[tuple[int, int]]:
    '''Return character spans (start, end) in the ORIGINAL text.'''

def redact(text: str, spans: list[tuple[int, int]]) -> str:
    '''Remove the spans after expanding each to its whole sentence.'''
"""

import os
import re
from typing import List, Optional, Tuple
import pysbd
import torch
from transformers import AutoTokenizer

from ..data.labeling import build_model_input, LABEL2ID, ID2LABEL

# Global cache for loaded prototype model & tokenizer
_GLOBAL_MODEL = None
_GLOBAL_TOKENIZER = None
_CALIBRATED_THRESHOLD = 0.35  # Tuned for recall on validation split
DEFAULT_FALLBACK_RATIO = 0.50  # Calibrated on val.jsonl (Pareto-optimal: block rate <30%, coverage >=99.9%)
DEFAULT_GAP_TOKENS = 2  # Bridge unflagged connective tokens inside injection
DEFAULT_CAP_RATIO = 2.5  # Relative expansion cap to prevent over-removal
BLOCKED_SENTINEL = "[BLOCKED: PROMPT INJECTION DETECTED]"


def set_inference_model(model, tokenizer, threshold: float = 0.35):
    """Set global model and tokenizer for predict_spans."""
    global _GLOBAL_MODEL, _GLOBAL_TOKENIZER, _CALIBRATED_THRESHOLD
    _GLOBAL_MODEL = model
    _GLOBAL_TOKENIZER = tokenizer
    _CALIBRATED_THRESHOLD = threshold


def load_prototype_model(
    checkpoint_dir: str = "checkpoints/prototype-token-head-deberta-v3-xsmall-20261110",
    backbone_name: str = "microsoft/deberta-v3-xsmall"
):
    """Load trained prototype checkpoint for inference."""
    global _GLOBAL_MODEL, _GLOBAL_TOKENIZER
    from ..models.token_head import SudarshanaTokenClassifier

    if os.path.exists(checkpoint_dir):
        tokenizer = AutoTokenizer.from_pretrained(backbone_name)
        model = SudarshanaTokenClassifier(backbone_name_or_path=backbone_name)
        weights_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        model.eval()
        set_inference_model(model, tokenizer)
        return model, tokenizer
    return None, None


def expand_span_structure_aware(
    text: str,
    span: Tuple[int, int],
    cap_ratio: float = DEFAULT_CAP_RATIO,
    min_cap_chars: int = 150
) -> Tuple[int, int]:
    """
    Expand a single character span to its enclosing structural unit:
    1. HTML comment <!-- ... -->
    2. Markdown code block ```...```
    3. Python docstring '''...''' or \"\"\"...\"\"\"
    4. Markdown table row | ... |
    5. Contiguous code comment block (# ... or // ...)
    6. Sentence / paragraph boundary bounded by newlines
    Enforces maximum expansion cap to prevent over-removal.
    """
    s_start, s_end = span
    span_len = s_end - s_start
    max_len = max(min_cap_chars, int(cap_ratio * span_len)) if cap_ratio < float("inf") else len(text)

    # 1. HTML comment <!-- ... -->
    left_chunk = text[:s_start]
    last_comment_open = left_chunk.rfind("<!--")
    if last_comment_open != -1:
        last_comment_close = left_chunk.rfind("-->")
        if last_comment_close < last_comment_open:
            comment_end = text.find("-->", s_end)
            if comment_end != -1:
                e_end = comment_end + 3
                if e_end < len(text) and text[e_end] == "\n":
                    e_end += 1
                e_start = last_comment_open
                if e_start > 0 and text[e_start - 1] == "\n":
                    e_start -= 1
                if (e_end - e_start) <= max_len:
                    return (e_start, e_end)

    # 2. Markdown code fence ``` ... ```
    fence_pattern = re.compile(r"```[^\n]*\n[\s\S]*?```")
    for m in fence_pattern.finditer(text):
        if m.start() <= s_start and m.end() >= s_end:
            e_start, e_end = m.start(), m.end()
            if e_start > 0 and text[e_start - 1] == "\n":
                e_start -= 1
            if e_end < len(text) and text[e_end] == "\n":
                e_end += 1
            if (e_end - e_start) <= max_len:
                return (e_start, e_end)

    # 3. Python docstring '''...''' or """..."""
    for q in ["'''", '"""']:
        doc_pattern = re.compile(re.escape(q) + r"[\s\S]*?" + re.escape(q))
        for m in doc_pattern.finditer(text):
            if m.start() <= s_start and m.end() >= s_end:
                e_start, e_end = m.start(), m.end()
                if e_start > 0 and text[e_start - 1] == "\n":
                    e_start -= 1
                if e_end < len(text) and text[e_end] == "\n":
                    e_end += 1
                if (e_end - e_start) <= max_len:
                    return (e_start, e_end)

    # 4. Markdown Table Row (| ... |)
    prev_nl = text.rfind("\n", 0, s_start)
    line_start = 0 if prev_nl == -1 else prev_nl + 1
    next_nl = text.find("\n", s_end)
    line_end = len(text) if next_nl == -1 else next_nl
    enclosing_line = text[line_start:line_end]

    if "|" in enclosing_line and (enclosing_line.strip().startswith("|") or enclosing_line.strip().endswith("|")):
        e_end = line_end + 1 if line_end < len(text) and text[line_end] == "\n" else line_end
        if (e_end - line_start) <= max_len:
            return (line_start, e_end)

    # 5. Contiguous code comment lines (# ... or // ...)
    if enclosing_line.strip().startswith("#") or enclosing_line.strip().startswith("//"):
        c_start = line_start
        cur_p = prev_nl
        while cur_p > 0:
            p_nl = text.rfind("\n", 0, cur_p)
            p_start = 0 if p_nl == -1 else p_nl + 1
            p_line = text[p_start:cur_p].strip()
            if p_line.startswith("#") or p_line.startswith("//"):
                c_start = p_start
                cur_p = p_nl
            else:
                break
        
        c_end = line_end
        cur_n = next_nl
        while cur_n != -1 and cur_n < len(text):
            n_nl = text.find("\n", cur_n + 1)
            n_end = len(text) if n_nl == -1 else n_nl
            n_line = text[cur_n + 1:n_end].strip()
            if n_line.startswith("#") or n_line.startswith("//"):
                c_end = n_end
                cur_n = n_nl
            else:
                break
        
        if c_end < len(text) and text[c_end] == "\n":
            c_end += 1
        if (c_end - c_start) <= max_len:
            return (c_start, c_end)

    # 6. Natural Language / Prose: Expand to sentence boundary within paragraph using PySBD
    seg = pysbd.Segmenter(language="en", clean=False)
    # Paragraph boundaries: find double newlines or single newlines
    prev_para = text.rfind("\n\n", 0, s_start)
    para_start = 0 if prev_para == -1 else prev_para + 2
    next_para = text.find("\n\n", s_end)
    para_end = len(text) if next_para == -1 else next_para

    para_text = text[para_start:para_end]
    sentences = seg.segment(para_text) if para_text.strip() else []

    if sentences:
        cursor = 0
        exp_s = s_start
        exp_e = s_end
        matched_any = False
        for s in sentences:
            idx = para_text.find(s, cursor)
            if idx != -1:
                abs_s = para_start + idx
                abs_e = abs_s + len(s)
                cursor = idx + len(s)
                if max(abs_s, s_start) < min(abs_e, s_end):
                    exp_s = min(exp_s, abs_s)
                    exp_e = max(exp_e, abs_e)
                    matched_any = True
            else:
                cursor = min(cursor + len(s), len(para_text))

        if matched_any:
            if (exp_e - exp_s) <= max_len:
                return (exp_s, exp_e)
            else:
                margin = (max_len - span_len) // 2
                return (max(0, s_start - margin), min(len(text), s_end + margin))

    return (s_start, s_end)


def expand_spans_to_sentences(
    text: str,
    spans: List[Tuple[int, int]],
    cap_ratio: float = DEFAULT_CAP_RATIO
) -> List[Tuple[int, int]]:
    """
    Structure-aware span expansion.
    Expands each flagged span to its enclosing structural unit (table row, code block,
    docstring, comment block, or natural sentence) with a relative expansion cap.
    Merges overlapping or contiguous boundaries.
    """
    if not spans or not text.strip():
        return []

    expanded = []
    for sp in spans:
        exp = expand_span_structure_aware(text, sp, cap_ratio=cap_ratio)
        expanded.append(exp)

    # Merge overlapping expanded intervals
    expanded.sort(key=lambda x: x[0])
    merged = [expanded[0]]
    for cur_start, cur_end in expanded[1:]:
        prev_start, prev_end = merged[-1]
        if cur_start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, cur_end))
        else:
            merged.append((cur_start, cur_end))

    return merged


def redact(
    text: str,
    spans: List[Tuple[int, int]],
    fallback_ratio: float = DEFAULT_FALLBACK_RATIO,
    blocked_sentinel: str = BLOCKED_SENTINEL
) -> str:
    """
    Remove the spans after expanding each to its whole sentence.
    Falls back to blocking if the flagged character share exceeds fallback_ratio.
    
    Args:
        text: Original document text.
        spans: List of character spans (start_char, end_char) to redact.
        fallback_ratio: Threshold above which document is completely blocked.
        blocked_sentinel: String returned when document is blocked.
        
    Returns:
        Redacted document text, or blocked sentinel.
    """
    if not spans or len(text.strip()) == 0:
        return text

    # Expand spans to full sentence boundaries
    expanded_spans = expand_spans_to_sentences(text, spans)
    if not expanded_spans:
        return text

    # Calculate total redacted character count
    total_redacted_chars = sum(end - start for start, end in expanded_spans)
    flagged_share = total_redacted_chars / max(len(text), 1)

    # Check fallback blocking condition
    if flagged_share >= fallback_ratio:
        return blocked_sentinel

    # Reconstruct document omitting expanded sentences
    result_pieces = []
    last_idx = 0
    for start, end in expanded_spans:
        result_pieces.append(text[last_idx:start])
        last_idx = end
    result_pieces.append(text[last_idx:])

    redacted_text = "".join(result_pieces).strip()
    return redacted_text


def predict_spans(
    text: str,
    user_task: Optional[str] = None
) -> List[Tuple[int, int]]:
    """
    Return character spans (start, end) in the ORIGINAL text.
    Runs sliding window inference, merges token predictions by highest probability,
    and returns detected injection spans.
    """
    global _GLOBAL_MODEL, _GLOBAL_TOKENIZER, _CALIBRATED_THRESHOLD

    if not text.strip():
        return []

    # Attempt to load checkpoint if not loaded
    if _GLOBAL_MODEL is None or _GLOBAL_TOKENIZER is None:
        load_prototype_model()

    if _GLOBAL_MODEL is None or _GLOBAL_TOKENIZER is None:
        # If model not available, return empty list
        return []

    tokenizer = _GLOBAL_TOKENIZER
    model = _GLOBAL_MODEL
    model.eval()

    text_to_tokenize = build_model_input(text, user_task)

    # Tokenize full text once
    full_enc = tokenizer(
        text_to_tokenize,
        truncation=False,
        return_offsets_mapping=True,
        return_tensors=None
    )

    all_input_ids = full_enc["input_ids"]
    all_attention_mask = full_enc["attention_mask"]
    all_offsets = full_enc["offset_mapping"]

    total_tokens = len(all_input_ids)
    max_length = 256
    stride = 64
    step = max_length - stride

    # Dictionary mapping token index -> max injection probability
    token_max_prob = [0.0] * total_tokens

    with torch.no_grad():
        for start_idx in range(0, total_tokens, step):
            end_idx = min(start_idx + max_length, total_tokens)
            win_ids = all_input_ids[start_idx:end_idx]
            win_mask = all_attention_mask[start_idx:end_idx]
            actual_len = len(win_ids)

            pad_len = max_length - actual_len
            if pad_len > 0:
                pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                win_ids = win_ids + [pad_id] * pad_len
                win_mask = win_mask + [0] * pad_len

            input_tensor = torch.tensor([win_ids], dtype=torch.long)
            mask_tensor = torch.tensor([win_mask], dtype=torch.long)

            outputs = model(input_ids=input_tensor, attention_mask=mask_tensor)
            logits = outputs["logits"][0][:actual_len]  # (actual_len, 3)
            probs = torch.softmax(logits, dim=-1)

            # Injection prob = P(B-INJ) + P(I-INJ)
            inj_probs = (probs[:, LABEL2ID["B-INJ"]] + probs[:, LABEL2ID["I-INJ"]]).tolist()

            for i, p in enumerate(inj_probs):
                global_token_idx = start_idx + i
                if p > token_max_prob[global_token_idx]:
                    token_max_prob[global_token_idx] = p

            if end_idx == total_tokens:
                break

    # Extract character spans from tokens exceeding calibrated threshold
    detected_char_spans = []
    in_span = False
    cur_span_start = 0
    cur_span_end = 0
    gap_count = 0

    for idx, p in enumerate(token_max_prob):
        char_start, char_end = all_offsets[idx]
        if char_start == char_end:  # special token
            continue

        if p >= _CALIBRATED_THRESHOLD:
            if not in_span:
                in_span = True
                cur_span_start = char_start
                cur_span_end = char_end
            else:
                cur_span_end = max(cur_span_end, char_end)
            gap_count = 0
        else:
            if in_span:
                gap_count += 1
                if gap_count <= DEFAULT_GAP_TOKENS:
                    # Bridge small subword gap inside injection
                    continue
                else:
                    detected_char_spans.append((cur_span_start, cur_span_end))
                    in_span = False
                    gap_count = 0

    if in_span:
        detected_char_spans.append((cur_span_start, cur_span_end))

    return detected_char_spans

