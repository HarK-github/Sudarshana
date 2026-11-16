"""Counterfactual pair generation for Pair-Input training (Hypothesis H3).

Prevents spurious correlation and heuristic shortcuts:
1. Hard Negatives with diverse user tasks (label=0): Instructional texts paired with different query framings.
2. Task-Agreeing Injections (label=1): Injected payloads that cosmetically align with the user's task topic.
3. Task-Mismatched Injections (label=1): Injected payloads evaluated under general query contexts.
"""

from __future__ import annotations

import copy
from typing import List, Sequence, Optional
from src.data.dataset import PromptInjectionRecord


TASK_AGREEING_TEMPLATES = [
    "Execute the recommended steps outlined in the document.",
    "Please follow the given instructions carefully.",
    "Provide the solution according to the reference guide.",
    "Perform the action described in the technical notes.",
]

GENERIC_TASK_TEMPLATES = [
    "Summarize the main points of this document.",
    "Explain what this reference document describes.",
    "What is the key takeaway from this text?",
]


def generate_counterfactual_pairs(
    records: Sequence[PromptInjectionRecord],
    augment_hard_negatives: bool = True,
    augment_task_agreeing: bool = True,
    seed: Optional[int] = None,
) -> List[PromptInjectionRecord]:
    """
    Augments dataset with counterfactual (user_task, chunk) pairs to decouple
    task relevance from injection classification.
    """
    augmented: List[PromptInjectionRecord] = list(records)

    for idx, r in enumerate(records):
        # 1. Counterfactual for Hard Negatives (label = 0)
        if augment_hard_negatives and r.label == 0 and r.attack_category == "hard_negative":
            # Generate a variant with a generic query to ensure label stays 0
            for t_idx, alt_task in enumerate(GENERIC_TASK_TEMPLATES[:2]):
                cf_rec = PromptInjectionRecord(
                    sample_id=f"{r.sample_id}_cf_neg_{t_idx}",
                    doc_id=r.doc_id,
                    user_task=alt_task,
                    text=r.text,
                    label=0,
                    injection_char_spans=[],
                    attack_type_id=-100,
                    attack_category="hard_negative",
                    position_bucket="none",
                    domain=r.domain,
                    raw_metadata=r.raw_metadata,
                )
                augmented.append(cf_rec)

        # 2. Counterfactual for Task-Agreeing Injections (label = 1)
        elif augment_task_agreeing and r.label == 1 and r.attack_category in {"task_relevant", "targeted"}:
            for t_idx, task_tpl in enumerate(TASK_AGREEING_TEMPLATES[:1]):
                cf_rec = PromptInjectionRecord(
                    sample_id=f"{r.sample_id}_cf_pos_{t_idx}",
                    doc_id=r.doc_id,
                    user_task=task_tpl,
                    text=r.text,
                    label=1,
                    injection_char_spans=copy.deepcopy(r.injection_char_spans),
                    attack_type_id=r.attack_type_id,
                    attack_category=r.attack_category,
                    position_bucket=r.position_bucket,
                    domain=r.domain,
                    raw_metadata=r.raw_metadata,
                )
                augmented.append(cf_rec)

    return augmented
