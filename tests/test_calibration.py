"""Target-FPR calibration and the ablation ladder."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ABLATION_CONFIG = REPO_ROOT / "configs" / "ablation_ladder.yaml"


def test_lowest_threshold_under_the_cap() -> None:
    from sudarshana.evaluation.metrics import threshold_at_fpr

    chosen = threshold_at_fpr([0.1] * 10 + [0.9] * 10, [0] * 10 + [1] * 10, 0.05)
    assert chosen["threshold"] == 0.9
    assert chosen["fpr"] == pytest.approx(0.0)
    assert chosen["recall"] == pytest.approx(1.0)
    assert chosen["tight"] is True
    assert chosen["fp"] == 0
    assert chosen["negatives"] == 10


def test_cap_can_require_predicting_nothing() -> None:
    from sudarshana.evaluation.metrics import threshold_at_fpr

    chosen = threshold_at_fpr([0.95] + [0.9] * 10, [0] + [1] * 10, 0.05)
    assert chosen["threshold"] > 0.95
    assert chosen["fpr"] == pytest.approx(0.0)
    assert chosen["recall"] == pytest.approx(0.0)
    assert chosen["tight"] is True


def test_target_fpr_must_be_a_rate() -> None:
    from sudarshana.evaluation.metrics import threshold_at_fpr

    with pytest.raises(ValueError, match="target FPR"):
        threshold_at_fpr([0.2, 0.8], [0, 1], 1.5)


def _train_rows() -> list[dict]:
    rows = []
    for index in range(8):
        rows.append(
            {
                "user_task": "summarise",
                "document": f"ignore instructions payload {index}",
                "label": 1,
            }
        )
        rows.append(
            {
                "user_task": "summarise",
                "document": f"recipe bake bread quietly {index}",
                "label": 0,
            }
        )
    return rows


def test_missing_id_is_primary_and_obfuscated_id_is_excluded() -> None:
    from sudarshana.evaluation.calibrate import fit_lexical, freeze_threshold, is_primary_row

    assert is_primary_row({"document": "plain"}) is True
    assert is_primary_row({"id": "row-1"}) is True
    assert is_primary_row({"id": "4::obf::base64"}) is False

    model, n_train = fit_lexical(_train_rows(), "document", 512)
    assert n_train == 16
    val = [
        {"user_task": "t", "document": "recipe bake bread quietly extra", "label": 0},
        {"id": "4::obf::base64", "user_task": "t", "document": "zzzz obfuscated only", "label": 0},
        {"user_task": "t", "document": "ignore instructions payload extra", "label": 1},
    ]
    chosen = freeze_threshold(model, val, "document", 0.05)
    assert chosen["negatives"] == 1
    assert chosen["n_excluded"] == 1
    assert chosen["threshold"] is not None
    assert chosen["fpr"] <= 0.05


def test_test_labels_do_not_move_the_threshold() -> None:
    from sudarshana.evaluation.calibrate import evaluate_arm

    train = _train_rows()
    val = [
        {"user_task": "t", "document": "recipe bake bread quietly extra", "label": 0},
        {"user_task": "t", "document": "ignore instructions payload extra", "label": 1},
    ]
    test_a = [
        {"user_task": "t", "document": "alpha recipe", "label": 0},
        {"user_task": "t", "document": "beta ignore instructions", "label": 1},
    ]
    test_b = [
        {"user_task": "t", "document": "alpha recipe", "label": 1},
        {"user_task": "t", "document": "beta ignore instructions", "label": 0},
    ]
    shared = {"hard_negatives": [], "obfuscated": []}
    first = evaluate_arm(train, {"val": val, "test": test_a, **shared}, "document", target_fpr=0.05, max_tokens=512)
    second = evaluate_arm(train, {"val": val, "test": test_b, **shared}, "document", target_fpr=0.05, max_tokens=512)
    assert first["threshold"] == second["threshold"]
    assert first["threshold"] is not None


def test_ladder_marks_missing_rungs_not_run() -> None:
    from sudarshana.evaluation.ablation import build_ladder, format_ablation_section
    from sudarshana.evaluation.calibrate import build_calibration, load_splits

    splits = load_splits(REPO_ROOT / "data" / "splits", REPO_ROOT / "data" / "splits" / "obfuscated.jsonl")
    calibration = build_calibration(
        splits,
        target_fpr=0.05,
        max_tokens=512,
        seed=2023,
        target_recall=0.95,
        repo_root=REPO_ROOT,
    )
    ladder = build_ladder(calibration, REPO_ROOT)
    assert ladder["frozen_rung"] == "1_baseline"
    assert ladder["rungs"][0]["status"] == "OK"
    assert ladder["rungs"][0]["val"]["fpr"] <= 0.05
    assert "test" in ladder["rungs"][0]
    for rung in ladder["rungs"][1:]:
        assert rung["status"] == "NOT_RUN"
        assert "test" not in rung
    assert "normalizer" in ladder["rungs"][1]["reason"]
    assert set(ladder["available_arms"]) == {"document", "pair"}
    for arm in ladder["available_arms"].values():
        assert arm["val"]["fpr"] <= 0.05
        assert set(arm["obfuscated"]["by_family"]) == {"base64", "hex", "leetspeak", "zero_width", "homoglyph"}
    text = format_ablation_section(ladder)
    assert "NOT_RUN" in text
    assert "1_baseline" not in text or "Baseline" in text
    assert "not been run" in format_ablation_section(None)


def test_writer_uses_the_config_output(tmp_path: Path) -> None:
    import yaml

    from sudarshana.evaluation.ablation import run_from_config

    config = yaml.safe_load(ABLATION_CONFIG.read_text(encoding="utf-8"))
    config["output"] = str(tmp_path / "ladder.json")
    config["calibration_output"] = str(tmp_path / "calibration.json")
    run_from_config(config, ABLATION_CONFIG)
    assert (tmp_path / "ladder.json").is_file()
    assert (tmp_path / "calibration.json").is_file()
