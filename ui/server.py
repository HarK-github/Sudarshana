"""Sudarshana Demo UI Server.

Standard-library-only HTTP server providing interactive inspection and redaction:
- REPLAY mode: 206 test records with recorded predictions from audit
- LIVE MODEL mode: runs DeBERTa token head if weight files exist
- HEURISTIC DEMO mode: regex & NFKC scanner when weights are absent
- RESULTS: headline tables from results/p5/report_facts.json

Usage:
    python ui/server.py [--port 8000] [--no-browser]
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import re
import sys
import unicodedata
import webbrowser
from http import HTTPStatus
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Attempt to load redaction engine
_REDACT_AVAILABLE = False
try:
    from sudarshana.redaction.redact import redact as _repo_redact, BLOCKED_SENTINEL, DEFAULT_FALLBACK_RATIO
    _REDACT_AVAILABLE = True
except Exception:
    BLOCKED_SENTINEL = "[BLOCKED: PROMPT INJECTION DETECTED]"
    DEFAULT_FALLBACK_RATIO = 0.50

    def _repo_redact(text: str, spans: List[Tuple[int, int]], fallback_ratio: float = 0.50) -> str:
        if not spans:
            return text
        spans = sorted(spans, key=lambda x: x[0])
        total_len = sum(e - s for s, e in spans)
        if total_len / max(1, len(text)) >= fallback_ratio:
            return BLOCKED_SENTINEL
        out = []
        last = 0
        for s, e in spans:
            out.append(text[last:s])
            out.append("[REDACTED]")
            last = e
        out.append(text[last:])
        return "".join(out)


def check_live_weights() -> bool:
    weight_patterns = [
        str(REPO_ROOT / "models" / "final_step6_augmented" / "**" / "*.safetensors"),
        str(REPO_ROOT / "models" / "final_step6_augmented" / "**" / "*.bin"),
        str(REPO_ROOT / "models" / "final_step6_augmented" / "**" / "*.pt"),
        str(REPO_ROOT / "checkpoints" / "**" / "*.safetensors"),
        str(REPO_ROOT / "checkpoints" / "**" / "*.bin"),
        str(REPO_ROOT / "checkpoints" / "**" / "*.pt"),
    ]
    for pattern in weight_patterns:
        if glob.glob(pattern, recursive=True):
            return True
    return False


HAS_LIVE_MODEL = check_live_weights()


def heuristic_detect_spans(text: str) -> List[Tuple[int, int]]:
    """Heuristic scanner for injection patterns when trained weights are absent."""
    if not text:
        return []
    spans: List[Tuple[int, int]] = []

    # 1. Zero-width character scan
    zw_pattern = re.compile(r"[\u200B-\u200D\uFEFF]+")
    for m in zw_pattern.finditer(text):
        spans.append((m.start(), m.end()))

    # 2. Homoglyph / NFKC normalization check
    normalized = unicodedata.normalize("NFKC", text)

    # 3. Instruction override & command injection regexes
    patterns = [
        re.compile(r"(?i)\b(ignore|disregard|forget|override|bypass)\b[\s\S]{0,50}\b(instructions?|prompt|system|previous|all)\b"),
        re.compile(r"(?i)\b(you\s+(must|should|are\s+now\s+required\s+to|have\s+to))\b"),
        re.compile(r"(?i)\b(send|exfiltrate|forward|post|fetch|curl|wget)\b[\s\S]{0,40}\b(https?://|[a-z0-9._%+-]+@[a-z0-9.-]+|webhook|server)\b"),
        re.compile(r"(?i)\b(act\s+as|pretend\s+to\s+be|you\s+are\s+a\s+helpful\s+assistant\s+that\s+ignores)\b"),
        re.compile(r"(?i)\b(new\s+task|system\s+instruction|developer\s+mode|jailbreak)\b"),
        re.compile(r"(?i)\b(do\s+not\s+(follow|obey|execute)\s+the\s+(original|user))\b"),
    ]

    for pat in patterns:
        for m in pat.finditer(text):
            # Expand to sentence/line boundaries
            start = text.rfind("\n", 0, m.start())
            start = 0 if start == -1 else start + 1
            end = text.find("\n", m.end())
            end = len(text) if end == -1 else end
            spans.append((start, end))

    # 4. Base64 payload scan
    b64_pattern = re.compile(r"\b[A-Za-z0-9+/]{24,}={0,2}\b")
    for m in b64_pattern.finditer(text):
        b64_str = m.group()
        try:
            decoded = base64.b64decode(b64_str).decode("utf-8", errors="ignore")
            if any(k in decoded.lower() for k in ["ignore", "instruction", "system", "prompt", "bypass", "override"]):
                spans.append((m.start(), m.end()))
        except Exception:
            pass

    if not spans:
        return []

    # Merge overlapping spans
    spans.sort(key=lambda x: x[0])
    merged = [spans[0]]
    for cur_s, cur_e in spans[1:]:
        prev_s, prev_e = merged[-1]
        if cur_s <= prev_e:
            merged[-1] = (prev_s, max(prev_e, cur_e))
        else:
            merged.append((cur_s, cur_e))
    return merged


class SudarshanaState:
    def __init__(self) -> None:
        self.samples: List[Dict[str, Any]] = []
        self.facts: Dict[str, Any] = {}
        self._load_samples()
        self._load_facts()

    def _load_samples(self) -> None:
        split_p = REPO_ROOT / "data" / "splits" / "test.jsonl"
        audit_p = REPO_ROOT / "reports" / "audit" / "raw_test_predictions.jsonl"

        records: List[Dict[str, Any]] = []
        if split_p.exists():
            with open(split_p, "r", encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]

        audit_data: List[Dict[str, Any]] = []
        if audit_p.exists():
            with open(audit_p, "r", encoding="utf-8") as f:
                audit_data = [json.loads(line) for line in f if line.strip()]

        for i, r in enumerate(records):
            p = audit_data[i] if i < len(audit_data) else {}
            golds = p.get("golds", [])
            preds = p.get("pred_spans", [])
            is_attacked = bool(r.get("label", 0) == 1 or len(golds) > 0)
            self.samples.append({
                "id": i,
                "label": 1 if is_attacked else 0,
                "attack_type": r.get("attack_type", "unknown"),
                "position": r.get("position", "none"),
                "domain": r.get("domain", "text"),
                "document": r.get("document", ""),
                "user_task": r.get("user_task", ""),
                "length": len(r.get("document", "")),
                "gold_spans": golds,
                "pred_spans": preds,
                "is_blocked": p.get("is_blocked", False),
                "legitimate_retained": p.get("legitimate_retained", 1.0 if not is_attacked else 0.0),
                "residual_payload": p.get("residual_payload", 0.0),
            })

    def _load_facts(self) -> None:
        p = REPO_ROOT / "results" / "p5" / "report_facts.json"
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                self.facts = json.load(f)


STATE = SudarshanaState()


class SudarshanaRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        ui_dir = str(REPO_ROOT / "ui")
        super().__init__(*args, directory=ui_dir, **kwargs)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        if url.path == "/" or url.path == "/index.html":
            self.path = "/index.html"
            return super().do_GET()

        if url.path == "/api/samples":
            self._handle_samples_list()
        elif url.path == "/api/sample":
            self._handle_single_sample(url.query)
        elif url.path == "/api/results":
            self._handle_results()
        else:
            return super().do_GET()

    def do_POST(self) -> None:
        url = urlparse(self.path)
        if url.path == "/api/analyze":
            self._handle_analyze()
        else:
            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def _send_json(self, data: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _handle_samples_list(self) -> None:
        sample_meta = [
            {
                "id": s["id"],
                "label": s["label"],
                "attack_type": s["attack_type"],
                "position": s["position"],
                "domain": s["domain"],
                "length": s["length"],
            }
            for s in STATE.samples
        ]
        self._send_json(sample_meta)

    def _handle_single_sample(self, query_string: str) -> None:
        params = parse_qs(query_string)
        id_str = params.get("id", [None])[0]
        if id_str is None or not id_str.isdigit():
            self._send_json({"error": "Missing or invalid id parameter"}, status=HTTPStatus.BAD_REQUEST)
            return
        idx = int(id_str)
        if idx < 0 or idx >= len(STATE.samples):
            self._send_json({"error": "Sample index out of range"}, status=HTTPStatus.NOT_FOUND)
            return

        s = STATE.samples[idx]
        doc = s["document"]
        pred_spans = s["pred_spans"]
        redacted_doc = _repo_redact(doc, pred_spans, fallback_ratio=DEFAULT_FALLBACK_RATIO)
        is_blocked = (redacted_doc == BLOCKED_SENTINEL)

        doc_len = max(len(doc), 1)
        removed_chars = doc_len if is_blocked else (doc_len - len(redacted_doc))
        removed_pct = round((removed_chars / doc_len) * 100.0, 2)
        retained_pct = round(s["legitimate_retained"] * 100.0, 2)

        data = {
            "id": s["id"],
            "label": s["label"],
            "attack_type": s["attack_type"],
            "position": s["position"],
            "domain": s["domain"],
            "document": doc,
            "user_task": s["user_task"],
            "gold_spans": s["gold_spans"],
            "pred_spans": pred_spans,
            "redacted_text": redacted_doc,
            "is_blocked": is_blocked,
            "removed_char_pct": removed_pct,
            "content_retained_pct": retained_pct,
            "mode": "REPLAY (from test audit)",
            "redact_engine": "sudarshana.redaction.redact" if _REDACT_AVAILABLE else "stdlib fallback",
        }
        self._send_json(data)

    def _handle_analyze(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > 1024 * 1024:
            self._send_json({"error": "Payload exceeds 1 MB limit"}, status=HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return

        body = self.rfile.read(content_length).decode("utf-8", errors="ignore")
        try:
            req = json.loads(body)
        except Exception:
            self._send_json({"error": "Invalid JSON"}, status=HTTPStatus.BAD_REQUEST)
            return

        text = req.get("text", "")
        user_task = req.get("user_task", "")

        spans: List[Tuple[int, int]] = []
        mode = "HEURISTIC DEMO"
        note = "heuristic demo detector, not the trained model"

        if HAS_LIVE_MODEL:
            try:
                os.environ["HF_HUB_OFFLINE"] = "1"
                from sudarshana.redaction.redact import predict_spans
                spans = predict_spans(text, user_task=user_task)
                mode = "LIVE MODEL"
                note = "live DeBERTa token classifier inference"
            except Exception as e:
                spans = heuristic_detect_spans(text)
                note = f"live model inference failed ({str(e)[:40]}); fell back to heuristic demo"
        else:
            spans = heuristic_detect_spans(text)

        redacted_doc = _repo_redact(text, spans, fallback_ratio=DEFAULT_FALLBACK_RATIO)
        is_blocked = (redacted_doc == BLOCKED_SENTINEL)
        doc_len = max(len(text), 1)
        removed_chars = doc_len if is_blocked else (doc_len - len(redacted_doc))
        removed_pct = round((removed_chars / doc_len) * 100.0, 2)
        retained_pct = 0.0 if is_blocked else round(max(0.0, 100.0 - removed_pct), 2)

        data = {
            "mode": mode,
            "mode_note": note,
            "is_attacked": len(spans) > 0,
            "pred_spans": spans,
            "gold_spans": [],
            "document": text,
            "user_task": user_task,
            "redacted_text": redacted_doc,
            "is_blocked": is_blocked,
            "removed_char_pct": removed_pct,
            "content_retained_pct": retained_pct,
            "redact_engine": "sudarshana.redaction.redact" if _REDACT_AVAILABLE else "stdlib fallback",
        }
        self._send_json(data)

    def _handle_results(self) -> None:
        facts = STATE.facts
        tables = {
            "ladder": facts.get("ladder_table_3_2", []),
            "obfuscation": facts.get("obfuscation_table_3_5", {}),
            "latency": facts.get("latency_table_3_8", {}),
            "redaction": facts.get("model_redaction", {}),
        }
        self._send_json(tables)


def run_server(port: int = 8000, open_browser: bool = True) -> None:
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, SudarshanaRequestHandler)
    url = f"http://127.0.0.1:{port}"
    print(f"Sudarshana Demo UI running at {url}")
    print(f"Weights present: {HAS_LIVE_MODEL} | Redaction engine: {_REDACT_AVAILABLE}")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sudarshana Demo UI Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = parser.parse_args()
    run_server(port=args.port, open_browser=not args.no_browser)
