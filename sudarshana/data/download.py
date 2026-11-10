"""Download published benchmark files into ``data/raw``."""

from __future__ import annotations

import urllib.request
from pathlib import Path

USER_AGENT = "sudarshana-data-build"


def download(url: str, dest: Path, timeout: int = 120) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    temporary = dest.with_suffix(dest.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    print(f"downloading {url}")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        temporary.write_bytes(response.read())
    temporary.replace(dest)
