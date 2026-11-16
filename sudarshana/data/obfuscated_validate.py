"""Checks for the obfuscated held-out split.

Injected children must reproduce the transform, keep the parent context, and
stay out of train and val. Clean rows are verbatim copies of test rows. A
clean test document, or a whole clean row, that also appears in another split
is a property of the existing split, so that overlap is counted and not rejected.
"""

from __future__ import annotations

from typing import Any

from sudarshana.data.obfuscation import (
    CLEAN_FAMILY,
    canonical_sha256,
    context_text,
    rewrite_document,
    span_pairs,
)


def validate_obfuscated_split(
    rows: list[dict[str, Any]],
    parents: list[dict[str, Any]],
    *,
    families: list[str],
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Raise ``ValueError`` when the derived split drifts or leaks.

    Returns counts the card and the builder print.
    """
    if not families:
        raise ValueError("obfuscated held-out needs at least one family")

    parent_hashes = [canonical_sha256(row) for row in parents]
    if len(set(parent_hashes)) != len(parent_hashes):
        raise ValueError("test parents do not have unique canonical hashes")
    by_hash = dict(zip(parent_hashes, parents))

    train_docs = {row["document"] for row in train_rows}
    val_docs = {row["document"] for row in val_rows}
    train_hashes = {canonical_sha256(row) for row in train_rows}
    val_hashes = {canonical_sha256(row) for row in val_rows}

    expected_ids: set[str] = set()
    for index, parent in enumerate(parents):
        label = int(parent["label"])
        if label == 0:
            expected_ids.add(f"{index}::obf::{CLEAN_FAMILY}")
        elif label == 1:
            if not span_pairs(parent.get("injection_span")):
                raise ValueError(f"injected parent {index} has no span")
            for family in families:
                expected_ids.add(f"{index}::obf::{family}")
        else:
            raise ValueError(f"parent {index} has label {parent['label']!r}")

    seen_ids: set[str] = set()
    counts: dict[str, int] = {CLEAN_FAMILY: 0}
    for family in families:
        counts[family] = 0
    unchanged: dict[str, int] = {family: 0 for family in families}
    clean_overlap = {"train": 0, "val": 0}

    for row in rows:
        row_id = str(row.get("id"))
        if row_id in seen_ids:
            raise ValueError(f"duplicate obfuscated id {row_id}")
        seen_ids.add(row_id)

        family = row.get("obfuscation_family")
        parent_index = row.get("parent_index")
        digest = row.get("parent_sha256")
        if not isinstance(parent_index, int) or not (0 <= parent_index < len(parents)):
            raise ValueError(f"{row_id} has parent_index {parent_index!r}")
        parent = parents[parent_index]
        if digest != parent_hashes[parent_index] or digest not in by_hash:
            raise ValueError(f"{row_id} parent_sha256 does not match test row {parent_index}")
        for key in ("user_task", "label", "attack_type", "position"):
            if key in parent and row.get(key) != parent.get(key):
                raise ValueError(f"{row_id} changed parent field {key}")

        label = int(parent["label"])
        if label == 0:
            if family != CLEAN_FAMILY or row_id != f"{parent_index}::obf::{CLEAN_FAMILY}":
                raise ValueError(f"clean parent {parent_index} must be copied once as {CLEAN_FAMILY}")
            if row["document"] != parent["document"] or row.get("injection_span") != parent.get("injection_span"):
                raise ValueError(f"{row_id} clean control is not a verbatim copy")
            counts[CLEAN_FAMILY] += 1
            if row["document"] in train_docs:
                clean_overlap["train"] += 1
            if row["document"] in val_docs:
                clean_overlap["val"] += 1
            continue

        if family not in families or row_id != f"{parent_index}::obf::{family}":
            raise ValueError(f"{row_id} is not an injected child of the requested families")
        pairs = span_pairs(parent.get("injection_span"))
        rewritten, new_pairs = rewrite_document(parent["document"], pairs, str(family))
        if row["document"] != rewritten or span_pairs(row.get("injection_span")) != new_pairs:
            raise ValueError(f"{row_id} document or spans do not match family {family}")
        if context_text(row["document"], new_pairs) != context_text(parent["document"], pairs):
            raise ValueError(f"{row_id} changed text outside the payload")
        if digest in train_hashes or digest in val_hashes:
            raise ValueError(f"{row_id} injected parent row is also in train or val")
        if row["document"] in train_docs or row["document"] in val_docs:
            raise ValueError(f"{row_id} injected document appears in train or val")
        counts[str(family)] += 1
        original_payloads = [parent["document"][start:end] for start, end in pairs]
        new_payloads = [row["document"][start:end] for start, end in new_pairs]
        if original_payloads == new_payloads:
            unchanged[str(family)] += 1

    if seen_ids != expected_ids:
        missing = sorted(expected_ids - seen_ids)[:5]
        extra = sorted(seen_ids - expected_ids)[:5]
        raise ValueError(f"obfuscated ids do not match the test parents (missing {missing}, extra {extra})")

    injected_parents = sum(1 for row in parents if int(row["label"]) == 1)
    clean_parents = sum(1 for row in parents if int(row["label"]) == 0)
    return {
        "n_records": len(rows),
        "n_injected_parents": injected_parents,
        "n_clean": clean_parents,
        "counts": counts,
        "unchanged_payloads": unchanged,
        "clean_document_overlap": clean_overlap,
    }
