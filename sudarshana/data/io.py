"""JSONL helpers for raw benchmarks and the unified split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(row)
    return rows


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            yield row


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file or a JSON list / JSON object-of-lists."""
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        if not all(isinstance(item, dict) for item in payload):
            raise ValueError(f"{path} list items must be objects")
        return payload
    if isinstance(payload, dict):
        if payload and all(isinstance(value, list) for value in payload.values()):
            keys = list(payload)
            length = len(payload[keys[0]])
            rows = []
            for index in range(length):
                rows.append({key: payload[key][index] for key in keys})
            return rows
        if all(isinstance(value, dict) for value in payload.values()):
            return list(payload.values())
    raise ValueError(f"{path} is not a JSONL file, a JSON list, or a JSON object")


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(as_text(item) for item in value)
    return str(value)
