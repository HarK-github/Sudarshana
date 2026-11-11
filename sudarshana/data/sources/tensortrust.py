"""TensorTrust hijacking and extraction rows -> unified examples.

The defender prompt is the user task. The untrusted player message is the
document. Hijacking attacks are prompt hijacks. Extraction-benchmark attacks
are the held-out attack type. A clean row uses the defender's access code as
the player message, one row per defense, so the benign message is not an
injection.

Expected files (downloaded from HumanCompatibleAI/tensor-trust-data when
missing):

    hijacking_robustness_dataset.jsonl
    extraction_robustness_dataset.jsonl

Columns used: sample_id, pre_prompt, post_prompt, attack, access_code.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from sudarshana.data.download import download
from sudarshana.data.io import read_jsonl
from sudarshana.data.schema import UnifiedExample

TENSORTRUST_FILES = {
    "hijacking_robustness_dataset.jsonl": (
        "https://raw.githubusercontent.com/HumanCompatibleAI/tensor-trust-data/"
        "main/benchmarks/hijacking-robustness/v1/hijacking_robustness_dataset.jsonl"
    ),
    "extraction_robustness_dataset.jsonl": (
        "https://raw.githubusercontent.com/HumanCompatibleAI/tensor-trust-data/"
        "main/benchmarks/extraction-robustness/v1/extraction_robustness_dataset.jsonl"
    ),
}

MESSAGE_PREFIX = "User message:\n"


def attack_family(filename: str, row: dict) -> str:
    if row.get("is_prompt_extraction") is True or "extract" in filename.lower():
        return "prompt_extraction"
    declared = row.get("attack_type")
    if declared:
        return str(declared)
    return "prompt_hijack"


def defense_key(pre_prompt: str, post_prompt: str) -> str:
    material = pre_prompt.strip() + "\n---\n" + post_prompt.strip()
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _user_task(pre_prompt: str, post_prompt: str) -> str:
    parts = [part.strip() for part in (pre_prompt, post_prompt) if part and part.strip()]
    return "\n".join(parts)


def ensure_tensortrust_files(raw_dir: Path, do_download: bool) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = [raw_dir / name for name in TENSORTRUST_FILES]
    missing = [path for path in paths if not path.is_file()]
    if not missing:
        return paths
    extras = sorted(raw_dir.glob("*.jsonl"))
    if extras and not do_download:
        return extras
    if not do_download:
        raise FileNotFoundError(
            "TensorTrust JSONL files are missing from "
            f"{raw_dir}. Re-run without --skip-download."
        )
    for name, url in TENSORTRUST_FILES.items():
        download(url, raw_dir / name)
    return [raw_dir / name for name in TENSORTRUST_FILES]


def load_tensortrust(raw_dir: Path, *, download_missing: bool = False) -> list[UnifiedExample]:
    paths = ensure_tensortrust_files(raw_dir, download_missing)
    examples: list[UnifiedExample] = []
    seen_clean: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            pre_prompt = str(row.get("pre_prompt") or "")
            post_prompt = str(row.get("post_prompt") or "")
            attack = str(row.get("attack") or "")
            access_code = str(row.get("access_code") or "")
            if attack.strip() == "":
                continue
            family = attack_family(path.name, row)
            sample_id = str(row.get("sample_id") or len(examples))
            user_task = _user_task(pre_prompt, post_prompt)
            key = defense_key(pre_prompt, post_prompt)
            document = MESSAGE_PREFIX + attack
            span = {"start": len(MESSAGE_PREFIX), "end": len(document)}
            examples.append(
                UnifiedExample(
                    user_task=user_task,
                    document=document,
                    label=1,
                    injection_span=[span],
                    attack_type=family,
                    id=f"tensortrust-{family}-{sample_id}",
                    source="tensortrust",
                    document_source="tensortrust",
                    attack_type_id=(
                        "tensortrust::prompt_extraction"
                        if family == "prompt_extraction"
                        else f"tensortrust::{family}::{key}"
                    ),
                    attack_category=family,
                    task_name="access_control",
                    modality="text",
                    position="user_message",
                    document_group=f"tensortrust:{key}",
                )
            )
            if key in seen_clean or access_code == "":
                continue
            seen_clean.add(key)
            examples.append(
                UnifiedExample(
                    user_task=user_task,
                    document=MESSAGE_PREFIX + access_code,
                    label=0,
                    injection_span=None,
                    attack_type="none",
                    id=f"tensortrust-clean-{sample_id}",
                    source="tensortrust",
                    document_source="tensortrust",
                    attack_type_id=f"tensortrust::none::{key}",
                    attack_category="none",
                    task_name="access_control",
                    modality="text",
                    position=None,
                    document_group=f"tensortrust:{key}",
                    reference_answer="Access Granted",
                )
            )
    return examples
