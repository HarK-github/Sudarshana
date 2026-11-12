"""The Day-2 field list is frozen in contract.json."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from sudarshana.data.schema import CLEAN_ATTACK_CATEGORIES, SCHEMA_VERSION, SOURCES, UNIFIED_FIELDS

CONTRACT = Path(__file__).resolve().parents[1] / "sudarshana" / "data" / "contract.json"


class ContractFreezeTests(unittest.TestCase):
    def test_contract_matches_code(self) -> None:
        payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], SCHEMA_VERSION)
        self.assertEqual(payload["schema_version"], "1.1")
        self.assertTrue(payload["frozen"])
        self.assertEqual(payload["fields"], list(UNIFIED_FIELDS))
        self.assertEqual(payload["sources"], list(SOURCES))
        self.assertEqual(payload["label_rules"]["0"]["attack_type"], "none")
        self.assertIsNone(payload["label_rules"]["0"]["injection_span"])
        self.assertEqual(
            payload["label_rules"]["0"]["attack_category"],
            list(CLEAN_ATTACK_CATEGORIES),
        )
        self.assertEqual(
            payload["fields"][:5],
            ["user_task", "document", "label", "injection_span", "attack_type"],
        )


if __name__ == "__main__":
    unittest.main()
