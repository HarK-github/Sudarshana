"""Span-label QA for the schema 1.1 splits, and the validation-split audit.

    python -m sudarshana.data.span_qa --processed-dir data/processed --report reports/span_label_qa.json

Character spans stay indexes into ``document``. Token labels follow
``align_spans_to_token_labels``: a word-initial token is positive only when
its start offset lies inside a span, the first such word of each span is
``B-INJ``, and later words are ``I-INJ``. Specials and subword continuations
are ``-100``. ``::counterfactual`` rows stay in the detection data and are
omitted from token loss.

AgentDojo payloads that YAML wrapped onto several lines are the same payload
when whitespace is collapsed. That is the match ``find_payload_spans`` records.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sudarshana.data.counterfactuals import COUNTERFACTUAL_SUFFIX, MISMATCHED_SUFFIX
from sudarshana.data.io import read_jsonl
from sudarshana.data.labeling import (
    LABEL2ID,
    PAD_LABEL_ID,
    align_spans_to_token_labels,
    coerce_spans,
    tokenize_and_align_document,
)
from sudarshana.data.schema import SIBLING_SUFFIXES, validate_example, validate_split
from sudarshana.data.sources.agentdojo import (
    HELD_OUT_SUITE,
    VAL_INJECTION_TASK_ID,
    VAL_USER_TASK_ID,
)
from sudarshana.data.sources.bipia import (
    CODE_TEST_NAMES,
    CODE_TRAIN_NAMES,
    TEXT_TEST_CATEGORY,
    TEXT_TRAIN_CATEGORY,
    VAL_ATTACK_NAMES,
)
from sudarshana.data.sources.tensortrust import MESSAGE_PREFIX
from sudarshana.data.splits import val_group_count

B_INJ = LABEL2ID["B-INJ"]
I_INJ = LABEL2ID["I-INJ"]
O_LABEL = LABEL2ID["O"]
POSITIVE = (B_INJ, I_INJ)

BIPIA_TRAIN_NAMES = (set(TEXT_TRAIN_CATEGORY) | set(CODE_TRAIN_NAMES)) - set(VAL_ATTACK_NAMES)
BIPIA_VAL_NAMES = set(VAL_ATTACK_NAMES)
BIPIA_TEST_NAMES = set(TEXT_TEST_CATEGORY) | set(CODE_TEST_NAMES)
BIPIA_NAMES = {
    "train": BIPIA_TRAIN_NAMES,
    "val": BIPIA_VAL_NAMES,
    "test": BIPIA_TEST_NAMES,
}
TEST_ATTACK_PREFIXES = ("bipia-text_test::", "bipia-code_test::")


def _whitespace_key(text: str) -> str:
    return " ".join(text.split())


def _span_dicts(row: dict[str, Any]) -> list[dict[str, int]]:
    raw = row.get("injection_span")
    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = [raw]
    return [{"start": int(item["start"]), "end": int(item["end"])} for item in raw]


class _Findings:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.hard: list[dict[str, Any]] = []
        self.warnings: list[dict[str, Any]] = []
        self.hard_count = 0
        self.warning_count = 0

    def add(
        self,
        severity: str,
        check: str,
        detail: str,
        *,
        row: dict[str, Any] | None = None,
        offsets: Any = None,
        record_id: str = "",
        source: str = "",
    ) -> None:
        item = {
            "severity": severity,
            "check": check,
            "id": record_id or ("" if row is None else str(row.get("id", ""))),
            "source": source or ("" if row is None else str(row.get("source", ""))),
            "detail": detail,
        }
        if offsets is not None:
            item["offsets"] = offsets
        if severity == "warning":
            self.warning_count += 1
            if len(self.warnings) < self.limit:
                self.warnings.append(item)
            return
        self.hard_count += 1
        if len(self.hard) < self.limit:
            self.hard.append(item)


def _is_sibling(row: dict[str, Any]) -> bool:
    record_id = str(row["id"])
    return record_id.endswith(SIBLING_SUFFIXES)


def _bipia_positive_split(row: dict[str, Any]) -> str:
    attack_id = str(row["attack_type_id"])
    if attack_id.startswith(TEST_ATTACK_PREFIXES):
        return "test"
    if row["attack_type"] in BIPIA_VAL_NAMES:
        return "val"
    return "train"


def _check_schema_and_spans(rows: list[dict[str, Any]], findings: _Findings) -> None:
    schema_ok = True
    for row in rows:
        try:
            validate_example(row)
        except ValueError as exc:
            schema_ok = False
            findings.add("hard", "schema", str(exc), row=row, offsets=row.get("injection_span"))
            continue
        _check_source_span(row, findings)

    by_id = {row["id"]: row for row in rows}
    for row in rows:
        suffix = next((item for item in SIBLING_SUFFIXES if str(row["id"]).endswith(item)), None)
        if suffix is None:
            continue
        parent = by_id.get(row["id"][: -len(suffix)])
        if (
            parent is None
            or parent["split"] != row["split"]
            or parent["document"] != row["document"]
        ):
            findings.add(
                "hard",
                "sibling",
                "sibling must keep the parent split and document",
                row=row,
            )
            continue
        if suffix == COUNTERFACTUAL_SUFFIX and (
            row["injection_span"] is not None or row["label"] != 0
        ):
            findings.add(
                "hard",
                "counterfactual",
                "counterfactual sibling keeps label 0 and a null span",
                row=row,
                offsets=row.get("injection_span"),
            )
        if suffix == MISMATCHED_SUFFIX and (
            row["injection_span"] is not None or row["label"] != 0
        ):
            findings.add(
                "hard",
                "mismatched_task",
                "mismatched sibling keeps label 0 and a null span",
                row=row,
                offsets=row.get("injection_span"),
            )

    if schema_ok:
        try:
            validate_split(rows)
        except ValueError as exc:
            findings.add("hard", "validate_split", str(exc))


def _check_source_span(row: dict[str, Any], findings: _Findings) -> None:
    if row["label"] != 1:
        return
    document = row["document"]
    spans = _span_dicts(row)
    if row["source"] == "bipia" and len(spans) == 1:
        start = spans[0]["start"]
        position = row.get("position")
        if position == "start" and start != 0:
            findings.add(
                "hard",
                "bipia_span",
                "start-position BIPIA span must begin at 0",
                row=row,
                offsets=spans,
            )
        elif position in ("end", "middle") and start > 0 and document[start - 1] != "\n":
            findings.add(
                "hard",
                "bipia_span",
                "end/middle BIPIA span must begin just after the inserted newline",
                row=row,
                offsets=spans,
            )
    elif row["source"] == "agentdojo":
        texts = {_whitespace_key(document[item["start"] : item["end"]]) for item in spans}
        if len(texts) != 1:
            findings.add(
                "hard",
                "agentdojo_span",
                "AgentDojo spans must be the same payload after whitespace is collapsed",
                row=row,
                offsets=spans,
            )
    elif row["source"] == "tensortrust":
        span = spans[0] if len(spans) == 1 else None
        if (
            span is None
            or not document.startswith(MESSAGE_PREFIX)
            or span["start"] != len(MESSAGE_PREFIX)
            or span["end"] != len(document)
        ):
            findings.add(
                "hard",
                "tensortrust_span",
                "TensorTrust span must cover the text after 'User message:\\n'",
                row=row,
                offsets=spans,
            )


def _check_splits(
    rows: list[dict[str, Any]],
    findings: _Findings,
    *,
    full_inventory: bool,
    tensortrust_val_fraction: float,
    hard_negative_val_fraction: float,
    held_out_hard_negative_category: str,
    held_out_suite: str,
    val_injection_task_id: str,
    val_user_task_id: str,
) -> dict[str, Any]:
    by_attack: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["label"] == 1:
            by_attack[str(row["attack_type_id"])].add(str(row["split"]))
    disjoint = True
    for attack_id, splits in sorted(by_attack.items()):
        if len(splits) > 1:
            disjoint = False
            findings.add(
                "hard",
                "attack_type_id",
                f"{attack_id} is in {sorted(splits)}",
                record_id=attack_id,
            )

    bipia_names = {
        split: {
            row["attack_type"]
            for row in rows
            if row["source"] == "bipia" and row["label"] == 1 and row["split"] == split
        }
        for split in ("train", "val", "test")
    }
    for split, expected in BIPIA_NAMES.items():
        got = bipia_names[split]
        extra = got - expected
        missing = expected - got
        if extra:
            findings.add(
                "hard",
                "bipia_names",
                f"{split} has BIPIA attack names outside the {split} inventory: {sorted(extra)}",
            )
        if full_inventory and missing:
            findings.add(
                "hard",
                "bipia_names",
                f"{split} is missing BIPIA attack names: {sorted(missing)}",
            )
        elif missing:
            findings.add(
                "warning",
                "bipia_names",
                f"{split} inventory is partial; missing {sorted(missing)}",
            )

    for row in rows:
        if _is_sibling(row):
            continue
        if row["source"] == "bipia" and row["label"] == 1:
            expected = _bipia_positive_split(row)
            if row["split"] != expected:
                findings.add(
                    "hard",
                    "bipia_split",
                    f"expected split {expected}",
                    row=row,
                )
        elif row["source"] == "agentdojo":
            if row["document_source"] == held_out_suite:
                expected = "test"
            elif row["label"] == 1:
                expected = "val" if row["attack_type"] == val_injection_task_id else "train"
            else:
                expected = "val" if row["task_name"] == val_user_task_id else "train"
            if row["split"] != expected:
                findings.add("hard", "agentdojo_split", f"expected split {expected}", row=row)
        elif row["source"] == "tensortrust":
            continue
        elif row["source"] == "hard_negative":
            continue

    hijack_groups = sorted(
        {
            row["document_group"]
            for row in rows
            if row["source"] == "tensortrust" and row["attack_type"] == "prompt_hijack"
        }
    )
    hijack_count = val_group_count(len(hijack_groups), tensortrust_val_fraction)
    hijack_val = set(hijack_groups[-hijack_count:]) if hijack_count else set()
    hijack_all = set(hijack_groups)
    for row in rows:
        if row["source"] != "tensortrust" or _is_sibling(row):
            continue
        if row["attack_type"] == "prompt_extraction":
            expected = "test"
        elif row["document_group"] in hijack_all:
            expected = "val" if row["document_group"] in hijack_val else "train"
        else:
            expected = "test"
        if row["split"] != expected:
            findings.add("hard", "tensortrust_split", f"expected split {expected}", row=row)

    hard_groups = sorted(
        {
            row["document_group"]
            for row in rows
            if row["source"] == "hard_negative"
            and row["document_source"] != held_out_hard_negative_category
        }
    )
    hard_count = val_group_count(len(hard_groups), hard_negative_val_fraction)
    hard_val = set(hard_groups[-hard_count:]) if hard_count else set()
    for row in rows:
        if row["source"] != "hard_negative" or _is_sibling(row):
            continue
        if row["document_source"] == held_out_hard_negative_category:
            expected = "test"
        elif row["document_group"] in hard_val:
            expected = "val"
        else:
            expected = "train"
        if row["split"] != expected:
            findings.add("hard", "hard_negative_split", f"expected split {expected}", row=row)

    train_groups = {
        row["document_group"]
        for row in rows
        if row["source"] == "bipia" and row["split"] == "train"
    }
    val_groups = {
        row["document_group"]
        for row in rows
        if row["source"] == "bipia" and row["split"] == "val"
    }
    overlap = sorted(train_groups & val_groups)
    if overlap:
        preview = ", ".join(overlap[:20])
        findings.add(
            "warning",
            "bipia_context_overlap",
            f"{len(overlap)} BIPIA document groups are in both train and val ({preview})",
        )
    return {
        "attack_type_id_disjoint": disjoint,
        "val_bipia_attack_names": sorted(bipia_names["val"]),
        "bipia_document_group_overlap_train_val": len(overlap),
    }


def _is_special(word_id: int | None, start: int, end: int) -> bool:
    return word_id is None or (start == 0 and end == 0)


def _check_token_labels(
    row: dict[str, Any],
    offsets: list[tuple[int, int]],
    word_ids: list[int | None],
    labels: list[int],
    text: str | None = None,
) -> list[str]:
    """Return hard-failure details for one tokenized row."""
    problems: list[str] = []
    spans = coerce_spans(row.get("injection_span"))
    if len(offsets) != len(word_ids) or len(offsets) != len(labels):
        return ["offset map, word ids, and labels differ in length"]

    seen_span: dict[int, int] = {}
    word_initial: list[tuple[int, int, int]] = []
    prev_word_id: int | None = None
    for index, ((start, end), word_id, label) in enumerate(zip(offsets, word_ids, labels)):
        if _is_special(word_id, start, end):
            if label != PAD_LABEL_ID:
                problems.append(f"special token at {(start, end)} has label {label}")
            continue
        if word_id == prev_word_id:
            if label != PAD_LABEL_ID:
                problems.append(f"subword continuation at {(start, end)} has label {label}")
            continue
        prev_word_id = word_id
        word_initial.append((index, start, end))
        owners = [span_index for span_index, span in enumerate(spans) if span[0] <= start < span[1]]
        if not owners:
            continue
        owner = owners[0]
        order = seen_span.get(owner, 0)
        expected = B_INJ if order == 0 else I_INJ
        seen_span[owner] = order + 1
        if label != expected:
            problems.append(
                f"span {spans[owner]} word {order} at {(start, end)} is {label}, expected {expected}"
            )

    promoted: dict[int, int] = {}
    if text is not None:
        for span_index, (span_start, span_end) in enumerate(spans):
            if seen_span.get(span_index, 0) > 0:
                continue
            for _index, start, end in word_initial:
                if start < span_start < end and text[start:span_start].strip() == "":
                    promoted[span_index] = _index
                    break

    for span_index, token_index in promoted.items():
        start, end = offsets[token_index]
        if labels[token_index] != B_INJ:
            problems.append(
                f"span {spans[span_index]} whitespace-prefixed token at {(start, end)} "
                f"is {labels[token_index]}, expected {B_INJ}"
            )
        seen_span[span_index] = seen_span.get(span_index, 0) + 1

    for token_index, start, end in word_initial:
        if labels[token_index] not in POSITIVE:
            continue
        inside = [span_index for span_index, span in enumerate(spans) if span[0] <= start < span[1]]
        if inside or token_index in promoted.values():
            continue
        problems.append(f"positive label {labels[token_index]} at {(start, end)} is outside every span")

    if row["label"] == 0:
        if any(label in POSITIVE for label in labels):
            problems.append("clean row has a B-INJ or I-INJ label")
        return problems

    for index, span in enumerate(spans):
        if seen_span.get(index, 0) == 0:
            problems.append(f"span {span} has no word-initial token whose start lies inside it")
    return problems


def _covered_length(ranges: list[tuple[int, int]]) -> int:
    if not ranges:
        return 0
    ordered = sorted(ranges)
    total = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start > current_end:
            total += current_end - current_start
            current_start, current_end = start, end
        elif end > current_end:
            current_end = end
    return total + (current_end - current_start)


def _span_coverage(spans: list[tuple[int, int]], token_ranges: list[tuple[int, int]]) -> float:
    span_total = sum(end - start for start, end in spans)
    if span_total <= 0:
        return 1.0
    covered = 0
    for start, end in spans:
        pieces = []
        for token_start, token_end in token_ranges:
            lo = max(start, token_start)
            hi = min(end, token_end)
            if lo < hi:
                pieces.append((lo, hi))
        covered += _covered_length(pieces)
    return covered / span_total


def _window_label_problem(
    document: str,
    spans: list[tuple[int, int]],
    tokenizer: Any,
    full_labels: list[int],
) -> str | None:
    windows = tokenize_and_align_document(document, spans, tokenizer)
    for window in windows:
        width = window["end_token_idx"] - window["start_token_idx"]
        sliced = full_labels[window["start_token_idx"] : window["end_token_idx"]]
        if window["labels"][:width] != sliced:
            return "window labels diverge from the full-sequence BIO labels"
    return None


def _align_rows(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    findings: _Findings,
    *,
    batch_size: int,
    max_chars: int,
    verify_windows: bool,
    coverage: list[tuple[str, float]] | None,
) -> int:
    checked = 0
    batch: list[dict[str, Any]] = []
    chars = 0

    def flush() -> None:
        nonlocal checked, chars
        if not batch:
            return
        encoded = tokenizer(
            [row["document"] for row in batch],
            truncation=False,
            padding=False,
            return_offsets_mapping=True,
            return_attention_mask=False,
        )
        for index, row in enumerate(batch):
            offsets = [tuple(pair) for pair in encoded["offset_mapping"][index]]
            word_ids = list(encoded.word_ids(index))
            spans = coerce_spans(row.get("injection_span"))
            labels = align_spans_to_token_labels(
                offsets, word_ids, spans, text=row["document"]
            )
            problems = _check_token_labels(
                row, offsets, word_ids, labels, text=row["document"]
            )
            if verify_windows:
                window_problem = _window_label_problem(row["document"], spans, tokenizer, labels)
                if window_problem:
                    problems.append(window_problem)
            for detail in problems:
                findings.add("hard", "token_alignment", detail, row=row, offsets=row.get("injection_span"))
            if coverage is not None and row["label"] == 1 and not str(row["id"]).endswith(COUNTERFACTUAL_SUFFIX):
                ranges = [
                    (start, end)
                    for (start, end), label in zip(offsets, labels)
                    if label in POSITIVE and end > start
                ]
                coverage.append((str(row.get("modality") or ""), _span_coverage(spans, ranges)))
            checked += 1
        batch.clear()
        chars = 0

    announce = len(rows) >= 1000
    for row in rows:
        size = len(row["document"])
        if batch and (len(batch) >= batch_size or chars + size > max_chars):
            flush()
            if announce and checked % 2000 < batch_size:
                print(f"aligned {checked} / {len(rows)} rows", flush=True)
        batch.append(row)
        chars += size
    flush()
    return checked


def _summarize(rows: list[dict[str, Any]], coverage: list[tuple[str, float]]) -> dict[str, Any]:
    counts: dict[str, Any] = {}
    for name in ("train", "val", "test"):
        subset = [row for row in rows if row["split"] == name]
        counts[name] = {
            "rows": len(subset),
            "positives": sum(row["label"] == 1 for row in subset),
            "negatives": sum(row["label"] == 0 for row in subset),
            "by_source": dict(Counter(row["source"] for row in subset)),
        }

    val_pos = [row for row in rows if row["split"] == "val" and row["label"] == 1]
    lengths = [item["end"] - item["start"] for row in val_pos for item in _span_dicts(row)]
    attack_types: dict[str, dict[str, int]] = {}
    for source in sorted({row["source"] for row in val_pos}):
        attack_types[source] = dict(
            Counter(row["attack_type"] for row in val_pos if row["source"] == source)
        )
    excluded = dict(
        Counter(row["split"] for row in rows if str(row["id"]).endswith(COUNTERFACTUAL_SUFFIX))
    )
    for name in ("train", "val", "test"):
        excluded.setdefault(name, 0)

    def _macro(items: list[float]) -> float | None:
        if not items:
            return None
        return round(sum(items) / len(items), 4)

    text_scores = [score for modality, score in coverage if modality == "text"]
    code_scores = [score for modality, score in coverage if modality == "code"]
    return {
        "counts": counts,
        "val_attack_types_by_source": attack_types,
        "span_stats_val": {
            "positive_rows": len(val_pos),
            "spans": len(lengths),
            "multi_span_rows": sum(len(_span_dicts(row)) > 1 for row in val_pos),
            "multi_span_rate": round(
                sum(len(_span_dicts(row)) > 1 for row in val_pos) / len(val_pos), 4
            )
            if val_pos
            else 0.0,
            "length_mean": round(statistics.fmean(lengths), 1) if lengths else 0.0,
            "length_median": statistics.median(lengths) if lengths else 0,
            "length_min": min(lengths) if lengths else 0,
            "length_max": max(lengths) if lengths else 0,
            "by_position": dict(Counter(str(row.get("position")) for row in val_pos)),
            "by_modality": dict(Counter(row["modality"] for row in val_pos)),
        },
        "token_coverage_val": {
            "macro": _macro([score for _, score in coverage]),
            "text": _macro(text_scores),
            "code": _macro(code_scores),
            "positive_rows": len(coverage),
            "text_rows": len(text_scores),
            "code_rows": len(code_scores),
        },
        "counterfactuals_excluded_from_token_loss": excluded,
    }


def run_span_qa(
    processed_dir: Path,
    *,
    tokenizer: Any = None,
    tokenizer_name: str = "microsoft/deberta-v3-xsmall",
    align: bool = True,
    train_sample_size: int = 500,
    seed: int = 2023,
    full_inventory: bool = True,
    tensortrust_val_fraction: float = 0.15,
    hard_negative_val_fraction: float = 0.15,
    held_out_hard_negative_category: str = "shell",
    held_out_suite: str = HELD_OUT_SUITE,
    val_injection_task_id: str = VAL_INJECTION_TASK_ID,
    val_user_task_id: str = VAL_USER_TASK_ID,
    batch_size: int = 8,
    max_chars: int = 100_000,
    failure_limit: int = 100,
) -> dict[str, Any]:
    processed_dir = Path(processed_dir)
    rows: list[dict[str, Any]] = []
    for name in ("train", "val", "test"):
        path = processed_dir / f"{name}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.extend(read_jsonl(path))

    findings = _Findings(failure_limit)
    _check_schema_and_spans(rows, findings)
    split_summary = _check_splits(
        rows,
        findings,
        full_inventory=full_inventory,
        tensortrust_val_fraction=tensortrust_val_fraction,
        hard_negative_val_fraction=hard_negative_val_fraction,
        held_out_hard_negative_category=held_out_hard_negative_category,
        held_out_suite=held_out_suite,
        val_injection_task_id=val_injection_task_id,
        val_user_task_id=val_user_task_id,
    )

    coverage: list[tuple[str, float]] = []
    aligned = {"val": 0, "train_sample": 0}
    if align:
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        val_rows = [row for row in rows if row["split"] == "val"]
        train_rows = [row for row in rows if row["split"] == "train"]
        sample_size = min(train_sample_size, len(train_rows))
        train_sample = random.Random(seed).sample(train_rows, sample_size) if sample_size else []
        aligned["val"] = _align_rows(
            val_rows,
            tokenizer,
            findings,
            batch_size=batch_size,
            max_chars=max_chars,
            verify_windows=False,
            coverage=coverage,
        )
        aligned["train_sample"] = _align_rows(
            train_sample,
            tokenizer,
            findings,
            batch_size=batch_size,
            max_chars=max_chars,
            verify_windows=True,
            coverage=None,
        )

    summary = _summarize(rows, coverage)
    report = {
        "processed_dir": processed_dir.as_posix(),
        "tuning_split": (processed_dir / "val.jsonl").as_posix(),
        "prototype_val_split": "data/splits/val.jsonl",
        "hard_failures": findings.hard_count,
        "warnings": findings.warning_count,
        "failures": findings.hard,
        "failures_truncated": findings.hard_count > len(findings.hard),
        "warning_details": findings.warnings,
        "warnings_truncated": findings.warning_count > len(findings.warnings),
        "aligned_rows": aligned,
        "full_inventory": full_inventory,
        **split_summary,
        **summary,
    }
    return report


def render_markdown(report: dict[str, Any]) -> str:
    counts = report["counts"]
    coverage = report["token_coverage_val"]
    spans = report["span_stats_val"]
    excluded = report["counterfactuals_excluded_from_token_loss"]
    lines = [
        "# Span-label QA and validation split",
        "",
        "Redaction threshold, `lambda_tok`, and false-positive calibration read "
        f"`{report['tuning_split']}`. The prototype file `{report['prototype_val_split']}` "
        "is the set that produced redaction threshold 0.35.",
        "",
        f"Hard failures: {report['hard_failures']}. Warnings: {report['warnings']}.",
        "",
        "## Counts",
        "",
        "| Split | Rows | Positives | Negatives |",
        "|---|---:|---:|---:|",
    ]
    for name in ("train", "val", "test"):
        item = counts[name]
        lines.append(f"| {name} | {item['rows']} | {item['positives']} | {item['negatives']} |")
    lines.extend(["", "### By source", ""])
    for name in ("train", "val", "test"):
        parts = ", ".join(f"{source} {count}" for source, count in sorted(counts[name]["by_source"].items()))
        lines.append(f"- **{name}**: {parts}")

    lines.extend(["", "## Validation attack types", ""])
    for source, names in report["val_attack_types_by_source"].items():
        listing = ", ".join(f"{attack} ({count})" for attack, count in sorted(names.items()))
        lines.append(f"- **{source}**: {listing}")
    lines.extend(
        [
            "",
            "Validation BIPIA names: " + ", ".join(report["val_bipia_attack_names"]) + ".",
            "Positive `attack_type_id` values are pairwise disjoint across train, val, and test."
            if report["attack_type_id_disjoint"]
            else "Positive `attack_type_id` values overlap across splits.",
            "",
            "## Spans on validation",
            "",
            f"- Positive rows: {spans['positive_rows']}",
            f"- Spans: {spans['spans']}",
            f"- Multi-span rows: {spans['multi_span_rows']} ({spans['multi_span_rate']})",
            f"- Span length mean / median / min / max: {spans['length_mean']} / {spans['length_median']} / {spans['length_min']} / {spans['length_max']}",
            f"- Position: {spans['by_position']}",
            f"- Modality: {spans['by_modality']}",
            "",
            "## Token coverage on validation",
            "",
            "Coverage is the share of span characters that sit inside a supervised `B-INJ` or `I-INJ` token, macro-averaged over positive validation rows. Continuation pieces are unsupervised, so the score stays below 1 when a word is split.",
            "",
            f"- All: {coverage['macro']} over {coverage['positive_rows']} rows",
            f"- Text: {coverage['text']} over {coverage['text_rows']} rows",
            f"- Code: {coverage['code']} over {coverage['code_rows']} rows",
            f"- Aligned rows: val {report['aligned_rows']['val']}, train sample {report['aligned_rows']['train_sample']}",
            "",
            "## Counterfactuals excluded from token loss",
            "",
            f"- Train: {excluded.get('train', 0)}",
            f"- Val: {excluded.get('val', 0)}",
            f"- Test: {excluded.get('test', 0)}",
            "",
            "These rows keep the parent document with label 0 and a null span, because the user task is the attack text. Detection training keeps them. Token supervision drops them.",
            "",
            "## BIPIA document groups in both train and val",
            "",
            f"{report['bipia_document_group_overlap_train_val']} groups. "
            "The clean copy of every 5th train-file context is validation, while injections of non-validation attack types on that context stay in train. Attack types stay disjoint.",
            "",
        ]
    )
    if report["failures"]:
        lines.extend(["## Hard failures", ""])
        for item in report["failures"]:
            lines.append(f"- `{item['check']}` {item['id']} ({item['source']}): {item['detail']}")
        lines.append("")
    if report["warning_details"]:
        lines.extend(["## Warnings", ""])
        for item in report["warning_details"]:
            lines.append(f"- `{item['check']}`: {item['detail']}")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit span labels and the validation split.")
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("reports/span_label_qa.json"))
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--tokenizer", default="microsoft/deberta-v3-xsmall")
    parser.add_argument("--train-sample", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--partial-inventory", action="store_true")
    parser.add_argument("--skip-align", action="store_true")
    parser.add_argument("--tensortrust-val-fraction", type=float, default=0.15)
    parser.add_argument("--hard-negative-val-fraction", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--failure-limit", type=int, default=100)
    args = parser.parse_args(argv)

    report = run_span_qa(
        args.processed_dir,
        tokenizer_name=args.tokenizer,
        align=not args.skip_align,
        train_sample_size=args.train_sample,
        seed=args.seed,
        full_inventory=not args.partial_inventory,
        tensortrust_val_fraction=args.tensortrust_val_fraction,
        hard_negative_val_fraction=args.hard_negative_val_fraction,
        batch_size=args.batch_size,
        failure_limit=args.failure_limit,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path = args.markdown if args.markdown is not None else args.report.with_suffix(".md")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(
        f"wrote {args.report} ({report['hard_failures']} hard failures, {report['warnings']} warnings)"
    )
    return 1 if report["hard_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
