from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from causal_observation_reproduction.experiments.one_step import (
    POSITIVE,
    build_one_step_world,
    evaluate_one_step_world,
    materialize_one_step_dataset,
    write_one_step_dataset,
)
from causal_observation_reproduction.experiments.one_step_learning import (
    FEATURE_NAMES,
    LABEL_FIELDS,
    OneStepLearningConfig,
    run_one_step_calibration,
    write_one_step_calibration,
)


def test_one_step_source_configuration_count_is_explicit() -> None:
    rows = materialize_one_step_dataset()

    assert len(rows) == 634
    assert sum(row.dataset_panel == "frozen_test" for row in rows) == 60
    assert sum(row.dataset_panel == "independent_confirmation" for row in rows) == 300
    assert (
        sum(
            row.dataset_panel not in {"frozen_test", "independent_confirmation"}
            for row in rows
        )
        == 274
    )


def test_one_step_golden_positive_world_matches_the_source_derived_values() -> None:
    result = evaluate_one_step_world(build_one_step_world(POSITIVE, 2001))

    assert result.coarse_value == pytest.approx(0.4414666684)
    assert result.fine_gross_value == pytest.approx(0.8497705868)
    assert result.exact_gross_gap == pytest.approx(0.4083039183)
    assert result.continuation_action_gap == pytest.approx(0.7763472016)


def test_one_step_golden_pathology_world_preserves_frozen_signal_controller() -> None:
    result = evaluate_one_step_world(build_one_step_world("pathology", 1001))

    assert result.fine_gross_value == pytest.approx(0.6273222824)
    assert result.exact_net_gap == pytest.approx(-0.1656149314)


def test_one_step_writer_labels_the_count_reconciliation_gate(tmp_path: Path) -> None:
    data_path, manifest_path = write_one_step_dataset(tmp_path)

    records = data_path.read_text(encoding="utf-8").splitlines()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(records) == 634
    assert manifest["frozen_test_row_count"] == 60
    assert manifest["independent_confirmation_row_count"] == 300
    assert manifest["learning_row_count"] == 274
    assert manifest["paper_design_status"] == "learning_count_reconciled"


def test_one_step_calibration_keeps_labels_out_of_features_and_frozen_panel_disjoint(
    tmp_path: Path,
) -> None:
    config = OneStepLearningConfig(
        max_epochs=2,
        early_stopping_patience=1,
        bootstrap_samples=4,
        robustness_seeds=(17, 29),
    )

    result = run_one_step_calibration(config)
    artifacts = write_one_step_calibration(tmp_path, config)

    assert not set(FEATURE_NAMES) & LABEL_FIELDS
    assert len(result.splits["frozen_test"]) == 60
    assert len(result.splits["independent_confirmation"]) == 300
    assert len(result.splits["train"]) + len(result.splits["validation"]) == 274
    assert set(result.splits["train"]).isdisjoint(result.splits["validation"])
    assert {metric.policy for metric in result.metrics} >= {
        "linear_value_model",
        "mlp_value_model",
        "gnn_coo",
        "oracle_coo",
    }
    assert {metric.evaluation_panel for metric in result.metrics} == {
        "historical_frozen_test",
        "independent_confirmation",
    }
    assert artifacts[0] == tmp_path / "one_step_examples.jsonl"
    assert (tmp_path / "one_step_calibration_manifest.json").exists()
    seed_summary = json.loads(
        (tmp_path / "one_step_training_seed_summary.json").read_text(encoding="utf-8")
    )
    assert seed_summary["training_seed_count"] == 2
