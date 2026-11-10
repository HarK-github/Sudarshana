"""Assign train, val, and test without shuffling rows.

Rules, applied in order:

BIPIA
    Test-file attack types stay in test. Test-file documents stay in test.
    One train attack type from each category block is validation
    (see ``VAL_ATTACK_NAMES``). Remaining train attack types are train.
    Clean copies of train documents stay in train, except every 5th train
    document, whose clean copy is validation.

AgentDojo
    The held-out suite (travel) is test, including every attack type in it.
    ``injection_task_0`` is validation in every other suite.
    ``user_task_0`` clean rows are validation in every other suite.
    Everything else is train.

TensorTrust
    ``prompt_extraction`` is the held-out attack type and is test.
    Hijacking defenses are sorted by defense id. A tail slice is validation
    and the rest is train. The clean access-code row follows its defense.
"""

from __future__ import annotations

from collections import Counter

from sudarshana.data.schema import UnifiedExample
from sudarshana.data.sources.agentdojo import HELD_OUT_SUITE, VAL_INJECTION_TASK_ID, VAL_USER_TASK_ID
from sudarshana.data.sources.bipia import VAL_ATTACK_NAMES


def val_group_count(group_count: int, fraction: float) -> int:
    if group_count < 2:
        return 0
    count = int(round(group_count * fraction))
    count = max(count, 1)
    return min(count, group_count - 1)


def _bipia_context_split(example: UnifiedExample) -> str:
    parts = example.document_group.split(":")
    if len(parts) < 4 or parts[0] != "bipia":
        raise ValueError(f"bad BIPIA document group {example.document_group}")
    return parts[2]


def _bipia_context_index(example: UnifiedExample) -> int:
    return int(example.document_group.split(":")[3])


def _assign_bipia(example: UnifiedExample, val_names: set[str]) -> str:
    context_split = _bipia_context_split(example)
    if context_split == "test":
        return "test"
    if example.label == 1 and example.attack_type in val_names:
        return "val"
    if example.label == 0 and _bipia_context_index(example) % 5 == 0:
        return "val"
    return "train"


def _assign_agentdojo(
    example: UnifiedExample,
    held_out_suite: str,
    val_injection_task_id: str,
    val_user_task_id: str,
) -> str:
    if example.document_source == held_out_suite:
        return "test"
    if example.label == 1 and example.attack_type == val_injection_task_id:
        return "val"
    if example.label == 0 and example.task_name == val_user_task_id:
        return "val"
    return "train"


def _hijack_val_groups(examples: list[UnifiedExample], fraction: float) -> set[str]:
    groups = sorted(
        {
            example.document_group
            for example in examples
            if example.source == "tensortrust" and example.attack_type == "prompt_hijack"
        }
    )
    count = val_group_count(len(groups), fraction)
    return set(groups[-count:]) if count else set()


def assign_splits(
    examples: list[UnifiedExample],
    *,
    val_attack_names: tuple[str, ...] = VAL_ATTACK_NAMES,
    held_out_suite: str = HELD_OUT_SUITE,
    val_injection_task_id: str = VAL_INJECTION_TASK_ID,
    val_user_task_id: str = VAL_USER_TASK_ID,
    tensortrust_val_fraction: float = 0.15,
) -> list[UnifiedExample]:
    val_names = set(val_attack_names)
    hijack_val = _hijack_val_groups(examples, tensortrust_val_fraction)
    hijack_groups = {
        example.document_group
        for example in examples
        if example.source == "tensortrust" and example.attack_type == "prompt_hijack"
    }
    for example in examples:
        if example.source == "bipia":
            example.split = _assign_bipia(example, val_names)
        elif example.source == "agentdojo":
            example.split = _assign_agentdojo(
                example, held_out_suite, val_injection_task_id, val_user_task_id
            )
        elif example.source == "tensortrust":
            if example.attack_type == "prompt_extraction":
                example.split = "test"
            elif example.document_group in hijack_groups:
                example.split = "val" if example.document_group in hijack_val else "train"
            else:
                example.split = "test"
        else:
            raise ValueError(f"unknown source {example.source}")
    return examples


def split_report(rows: list[dict]) -> dict:
    def counts(key: str) -> dict[str, int]:
        return dict(Counter(row[key] for row in rows))

    by_split: dict[str, dict[str, int]] = {}
    for name in ("train", "val", "test"):
        subset = [row for row in rows if row["split"] == name]
        by_split[name] = {
            "rows": len(subset),
            "positives": sum(row["label"] == 1 for row in subset),
            "negatives": sum(row["label"] == 0 for row in subset),
            "by_source": dict(Counter(row["source"] for row in subset)),
            "by_attack_type": dict(Counter(row["attack_type"] for row in subset if row["label"] == 1)),
            "by_document_source": dict(Counter(row["document_source"] for row in subset)),
        }
    return {
        "rows": len(rows),
        "by_source": counts("source"),
        "by_split": by_split,
    }
