"""Benign instructional documents used as hard negatives.

Categories are recipes, tutorials, READMEs, shell pages, and legitimate
instruction emails. Every row is label 0. Shell pages are the published
tldr-pages texts. The other downloaded categories come from the pinned
manifest. Emails are the committed catalog unless a raw file is already
present.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import quote

import yaml

from sudarshana.data.download import download
from sudarshana.data.io import read_jsonl, write_jsonl
from sudarshana.data.schema import UnifiedExample

CATEGORIES = ("recipe", "tutorial", "readme", "shell", "email")
TLDR_ZIP = "https://codeload.github.com/tldr-pages/tldr/zip/refs/heads/main"
WIKIBOOKS_API = (
    "https://en.wikibooks.org/w/api.php?action=query&prop=extracts&explaintext=1"
    "&redirects=1&format=json&titles="
)


def _catalog_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "catalog"


def _manifest_path(override: Path | None) -> Path:
    if override is not None:
        return override
    return Path(__file__).resolve().parents[3] / "configs" / "hard_negatives.yaml"


def _strip_front_matter(text: str) -> str:
    if not text.startswith("---"):
        return text.strip()
    end = text.find("\n---", 3)
    if end == -1:
        return text.strip()
    return text[end + 4 :].strip()


def _cap(rows: list[dict[str, Any]], max_per_category: int | None) -> list[dict[str, Any]]:
    if max_per_category is None:
        return rows
    return rows[: max_per_category]


def _example(raw: dict[str, Any]) -> UnifiedExample:
    category = str(raw["category"])
    if category not in CATEGORIES:
        raise ValueError(f"unknown hard-negative category {category!r}")
    doc_id = str(raw["doc_id"])
    document = str(raw["document"]).strip()
    user_task = str(raw["user_task"]).strip()
    if not document or not user_task:
        raise ValueError(f"hard negative {category}/{doc_id} needs user_task and document")
    reference = raw.get("reference_answer")
    return UnifiedExample(
        user_task=user_task,
        document=document,
        label=0,
        injection_span=None,
        attack_type="none",
        id=f"hard-negative-{category}-{doc_id}",
        source="hard_negative",
        document_source=category,
        attack_type_id=f"hard_negative:{category}:{doc_id}:clean",
        attack_category="hard_negative",
        task_name=category,
        modality="text",
        position=None,
        document_group=f"hard_negative:{category}:{doc_id}",
        reference_answer=None if reference is None else str(reference),
    )


def _provenance(raw: dict[str, Any], example: UnifiedExample) -> dict[str, Any]:
    return {
        "id": example.id,
        "category": example.document_source,
        "license": raw.get("license"),
        "origin": raw.get("origin"),
    }


def _load_category_file(path: Path, max_per_category: int | None) -> tuple[list[UnifiedExample], list[dict]]:
    examples: list[UnifiedExample] = []
    provenance: list[dict] = []
    for raw in _cap(read_jsonl(path), max_per_category):
        example = _example(raw)
        examples.append(example)
        provenance.append(_provenance(raw, example))
    if not examples:
        raise ValueError(f"{path} has no hard negatives")
    return examples, provenance


def _write_records(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        raise RuntimeError(f"no hard-negative documents produced for {path.name}")
    write_jsonl(path, records)


def _tldr_command(path: str, text: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return Path(path).stem


def _fetch_tldr(dest: Path, max_per_category: int | None) -> None:
    archive = dest.parent / "tldr.zip"
    download(TLDR_ZIP, archive, timeout=180)
    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive) as bundle:
        names = sorted(
            name
            for name in bundle.namelist()
            if name.endswith(".md") and ("/pages/common/" in name or "/pages/linux/" in name)
        )
        limit = len(names) if max_per_category is None else max_per_category
        for name in names[:limit]:
            text = _strip_front_matter(bundle.read(name).decode("utf-8", errors="replace"))
            command = _tldr_command(name, text)
            doc_id = Path(name).stem
            records.append(
                {
                    "category": "shell",
                    "doc_id": doc_id,
                    "user_task": f"Show how to use {command}.",
                    "document": text,
                    "license": "MIT",
                    "origin": f"https://github.com/tldr-pages/tldr/blob/main/{name.split('/', 1)[-1]}",
                }
            )
    _write_records(dest, records)


def _fetch_text(url: str) -> str:
    import urllib.error
    import urllib.request

    from sudarshana.data.download import USER_AGENT

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"failed to download {url}: {exc}") from exc


def _fetch_wikibooks(title: str) -> str:
    payload = json.loads(_fetch_text(WIKIBOOKS_API + quote(title)))
    pages = payload.get("query", {}).get("pages", {})
    for page in pages.values():
        extract = page.get("extract")
        if extract and "missing" not in page:
            return str(extract).strip()
    raise RuntimeError(f"Wikibooks page {title!r} has no extract")


def _recipe_title(entry: dict[str, Any]) -> str:
    title = str(entry.get("title") or entry["id"])
    if title.startswith("Cookbook:"):
        title = title.split(":", 1)[1]
    return title.replace("_", " ")


def _tutorial_task(title: str) -> str:
    stripped = title.strip()
    if stripped.lower().startswith("how "):
        return stripped if stripped.endswith("?") else stripped + "?"
    return f"How do I {stripped[0].lower() + stripped[1:]}?"


def _fetch_manifest_category(category: str, entries: list[dict[str, Any]], dest: Path, max_per_category: int | None) -> None:
    records: list[dict[str, Any]] = []
    for entry in entries[: len(entries) if max_per_category is None else max_per_category]:
        fmt = str(entry.get("format") or "text")
        if fmt == "mediawiki":
            document = _fetch_wikibooks(str(entry["title"]))
            user_task = f"How do I make {_recipe_title(entry)}?"
        else:
            document = _strip_front_matter(_fetch_text(str(entry["url"])))
            if category == "readme":
                user_task = "Follow the setup instructions."
            else:
                user_task = _tutorial_task(str(entry.get("title") or entry["id"]))
        if not document.strip():
            raise RuntimeError(f"{category} entry {entry.get('id')} downloaded empty")
        records.append(
            {
                "category": category,
                "doc_id": str(entry["id"]),
                "user_task": user_task,
                "document": document.strip(),
                "license": entry.get("license"),
                "origin": entry.get("origin") or entry.get("url"),
            }
        )
    _write_records(dest, records)


def _copy_catalog_emails(dest: Path) -> None:
    catalog = _catalog_dir() / "legitimate_emails.jsonl"
    if not catalog.is_file():
        raise FileNotFoundError(f"legitimate email catalog is missing: {catalog}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(catalog.read_text(encoding="utf-8"), encoding="utf-8")


def ensure_hard_negative_files(
    raw_dir: Path,
    *,
    do_download: bool,
    max_per_category: int | None,
    manifest_path: Path | None = None,
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    missing = [category for category in CATEGORIES if not (raw_dir / f"{category}.jsonl").is_file()]
    if not missing:
        return
    if "email" in missing and (_catalog_dir() / "legitimate_emails.jsonl").is_file():
        _copy_catalog_emails(raw_dir / "email.jsonl")
        missing = [category for category in missing if category != "email"]
    if not missing:
        return
    if not do_download:
        raise FileNotFoundError(
            "Hard-negative files missing from "
            f"{raw_dir}: {', '.join(missing)}. Re-run without --skip-download."
        )
    manifest_file = _manifest_path(manifest_path)
    manifest = yaml.safe_load(manifest_file.read_text(encoding="utf-8")) or {}
    if "shell" in missing:
        _fetch_tldr(raw_dir / "shell.jsonl", max_per_category)
    for category in ("readme", "tutorial", "recipe"):
        if category not in missing:
            continue
        entries = manifest.get(category) or []
        if not entries:
            raise ValueError(f"{manifest_file} has no {category} entries")
        _fetch_manifest_category(category, entries, raw_dir / f"{category}.jsonl", max_per_category)


def load_hard_negatives(
    raw_dir: Path,
    *,
    download_missing: bool = False,
    max_per_category: int | None = 400,
    manifest_path: Path | None = None,
) -> tuple[list[UnifiedExample], list[dict[str, Any]]]:
    ensure_hard_negative_files(
        raw_dir,
        do_download=download_missing,
        max_per_category=max_per_category,
        manifest_path=manifest_path,
    )
    examples: list[UnifiedExample] = []
    provenance: list[dict[str, Any]] = []
    for category in CATEGORIES:
        path = raw_dir / f"{category}.jsonl"
        if not path.is_file() and category == "email":
            path = _catalog_dir() / "legitimate_emails.jsonl"
        loaded, notes = _load_category_file(path, max_per_category)
        examples.extend(loaded)
        provenance.extend(notes)
    return examples, provenance
