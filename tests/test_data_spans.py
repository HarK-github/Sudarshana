"""Schema 1.1 span recovery and attack-type split checks on the fixture corpus."""

import json

import pytest

from sudarshana.data.labeling import coerce_spans
from tests.support import build_fixture_corpus


@pytest.fixture(scope="module")
def processed_dir(tmp_path_factory):
    return build_fixture_corpus(tmp_path_factory.mktemp("processed"))


def _rows(processed_dir, name):
    path = processed_dir / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_span_roundtrip_verification(processed_dir):
    """document[start:end] recovers a non-empty injection on every positive row."""
    for name in ("train", "val", "test"):
        injected = [row for row in _rows(processed_dir, name) if row["label"] == 1]
        assert injected, f"{name} has no positive rows"
        for row in injected:
            spans = coerce_spans(row["injection_span"])
            assert spans, f"{row['id']} is positive with no span"
            for start, end in spans:
                assert 0 <= start < end <= len(row["document"])
                assert row["document"][start:end].strip()


def test_schema_compliance(processed_dir):
    """Clean rows use a null span. Positive rows use {start, end} dicts."""
    required = {"user_task", "document", "label", "injection_span", "attack_type"}
    for name in ("train", "val", "test"):
        for row in _rows(processed_dir, name):
            missing = required - set(row)
            assert not missing, f"{row.get('id')} missing {missing}"
            assert row["label"] in (0, 1)
            assert isinstance(row["document"], str)
            assert isinstance(row["user_task"], str)
            if row["label"] == 0:
                assert row["injection_span"] is None
                assert row["attack_type"] == "none"
            else:
                assert isinstance(row["injection_span"], list)
                for span in row["injection_span"]:
                    assert set(span) >= {"start", "end"}


def test_no_attack_type_leakage(processed_dir):
    """Positive attack_type_id values stay in one split."""
    by_split = {}
    for name in ("train", "val", "test"):
        by_split[name] = {
            row["attack_type_id"]
            for row in _rows(processed_dir, name)
            if row["label"] == 1
        }
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    for left, right in pairs:
        overlap = by_split[left] & by_split[right]
        assert not overlap, f"{left}/{right} share attack types {overlap}"
