"""Parity-oriented tests for the source-derived manuscript implementation."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from causal_observation_reproduction.reference.compute_characterization import (
    _gate_work,
    _voi_work,
)
from causal_observation_reproduction.reference.minigrid_audit import (
    MiniGridRelation,
    load_minigrid_external_control,
    summarize_minigrid_external_control,
)
from causal_observation_reproduction.reference.parity import (
    verify_one_step_calibration,
    verify_online_compute_counts,
)
from causal_observation_reproduction.reference.runners import run_sequential_smoke
from causal_observation_reproduction.reference.sequential_contract import (
    INFORMATION_ARMS,
    smoke_contract,
    validate_smoke_manifest,
)
from causal_observation_reproduction.reference.sequential_experiment import (
    ConditionalEfficiencyConfirmationConfig,
)

PROJECT_ROOT = Path(__file__).parents[1]


def _json_value(value: object) -> object:
    return json.loads(json.dumps(value, sort_keys=True))


def test_checked_sequential_config_equals_source_derived_defaults() -> None:
    sequential = json.loads(
        (PROJECT_ROOT / "configs/reference/sequential_confirmation.json").read_text(
            encoding="utf-8"
        )
    )
    assert sequential == _json_value(asdict(ConditionalEfficiencyConfirmationConfig()))


def test_minigrid_rows_recompute_every_appendix_aggregate() -> None:
    rows = load_minigrid_external_control()
    summaries = {
        summary.relation: summary
        for summary in summarize_minigrid_external_control(rows)
    }

    assert len(rows) == 16
    assert summaries[MiniGridRelation.IDENTITY].transferred == 4
    assert summaries[MiniGridRelation.SEED_SHIFT].transferred == 4
    assert summaries[MiniGridRelation.SCALE_CHANGE].transferred == 4
    assert summaries[MiniGridRelation.NEGATIVE_CONTROL].transferred == 0
    assert summaries[MiniGridRelation.IDENTITY].mean_readiness == pytest.approx(1.0)
    assert summaries[MiniGridRelation.SCALE_CHANGE].mean_readiness == pytest.approx(
        0.7811236821
    )
    assert summaries[MiniGridRelation.NEGATIVE_CONTROL].mean_readiness == pytest.approx(
        0.2653192179
    )


def test_source_defined_compute_counts_separate_gate_and_rollout_work() -> None:
    gate = _gate_work(
        ({"probe_count": 1, "horizon": 3}, {"probe_count": 0, "horizon": 5})
    )
    voi = _voi_work(
        (
            {
                "distractor_count": 2,
                "rollout_count": 4,
                "horizon": 3,
                "ambient_branch_count": 2,
            },
        )
    )

    assert gate == {
        "decision_invocations": 8,
        "threshold_table_lookups": 6,
        "scalar_threshold_comparisons": 6,
        "public_rollouts": 0,
        "simulated_control_transitions": 0,
        "candidate_branches": 0,
    }
    assert voi["public_rollouts"] == 4
    assert voi["simulated_control_transitions"] == 12
    assert voi["ambient_observation_scalar_elements"] == 4 * (2 * 3 + 1) * (8 + 2)
    assert voi["candidate_branches"] == 2


def test_online_compute_golden_check_is_fail_closed() -> None:
    payload = json.loads(
        (
            PROJECT_ROOT
            / "src/causal_observation_reproduction/reference/data/online_compute_reference.json"
        ).read_text(encoding="utf-8")
    )
    verify_online_compute_counts(payload)
    payload["counted_online_work"]["learned_causal_quotient"][
        "threshold_table_lookups"
    ] += 1
    with pytest.raises(ValueError, match="differ"):
        verify_online_compute_counts(payload)


def test_online_compute_golden_check_accepts_only_non_applicable_zero_fields() -> None:
    payload = json.loads(
        (
            PROJECT_ROOT
            / "src/causal_observation_reproduction/reference/data/online_compute_reference.json"
        ).read_text(encoding="utf-8")
    )
    generic = payload["counted_online_work"]["generic_bounded_rollout_voi"]
    generic["threshold_table_lookups"] = 0
    generic["scalar_threshold_comparisons"] = 0
    verify_online_compute_counts(payload)

    generic["threshold_table_lookups"] = 1
    with pytest.raises(ValueError, match="differ"):
        verify_online_compute_counts(payload)


def test_one_step_parity_preserves_the_known_corpus_disproof(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.json"
    split = tmp_path / "split.json"
    metrics = tmp_path / "metrics.csv"
    dataset.write_text(
        json.dumps(
            {
                "source_revision": "b4c468ef4dfac5eeab8b11a9e7f990ba36824748",
                "row_count": 274,
            }
        ),
        encoding="utf-8",
    )
    frozen = [
        f"frozen_test:{motif}:seed-{seed}"
        for motif in (
            "decision-relevant-refinement",
            "no-value-refinement",
            "pathology-failure-region",
        )
        for seed in range(20)
    ]
    split.write_text(
        json.dumps(
            {
                "splits": {
                    "train": [f"train:{index}" for index in range(144)],
                    "validation": [f"validation:{index}" for index in range(36)],
                    "frozen_test": frozen,
                }
            }
        ),
        encoding="utf-8",
    )
    metric_rows = [
        {
            "policy": policy,
            "eiv": eiv,
            "mean_regret": regret,
            "activation_rate": activation,
            "mae": mae,
        }
        for policy, eiv, regret, activation, mae in (
            ("never_refine", 0.0, 0.1, 0.0, 1.0),
            ("always_refine", 0.08, 0.05, 1.0, 0.9),
            ("linear_value_model", 0.14, 0.0, 2 / 3, 0.02),
            ("formula_coo_oracle_statistic", 0.08, 0.05, 2 / 3, 0.4),
            ("oracle_coo", 0.14, 0.0, 1 / 3, 0.0),
            ("mlp_value_model", 0.14, 0.0, 2 / 3, 0.08),
            ("gnn_coo", 0.14, 0.0, 2 / 3, 0.21),
            ("budget_matched_random", 0.07, 0.06, 0.68, 0.93),
        )
    ]
    with metrics.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    report = verify_one_step_calibration(
        dataset_manifest_path=dataset,
        split_manifest_path=split,
        policy_metrics_path=metrics,
    )

    assert report.metric_conclusions_match is True
    assert report.corpus_cardinality_matches is False
    assert report.development_rows == 180
    assert report.historical_rows == 34
    assert report.frozen_test_rows == 60


def test_sequential_contract_and_smoke_runner_use_the_exact_runtime(
    tmp_path: Path,
) -> None:
    manifest = {
        "schema_version": "causal_observation_reproduction.sequential_poscm_manifest.v1",
        "information_arms": list(INFORMATION_ARMS),
        "episode": {
            "horizon": 3,
            "information_actions": ["none", "refine", "probe"],
            "immediate_information_reward": 0.0,
        },
        "validity_gates": {
            "paired_latent_worlds": "required",
            "delayed_continuation_value": "required",
            "complete_feedback_boundary": "required",
            "outcome_blind_regime_assignment": "required",
        },
    }
    validate_smoke_manifest(manifest)
    contract = smoke_contract(5)
    assert contract["matched_latent_worlds"] is True
    assert contract["delayed_value_supported"] is True
    assert contract["feedback_leakage_detected"] is True
    assert contract["arms"] == list(INFORMATION_ARMS)

    artifacts = run_sequential_smoke(tmp_path)
    rows_path = next(
        path for path in artifacts if path.name == "confirmation_rows.json"
    )
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    assert len(rows) == 80
    assert {row["regime"] for row in rows} == {"positive", "null"}


def test_provenance_manifest_points_only_to_present_public_files() -> None:
    manifest_path = PROJECT_ROOT / "provenance/source_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["source_revision"] == "b4c468ef4dfac5eeab8b11a9e7f990ba36824748"
    assert len(payload["python_modules"]) == 19
    for entry in payload["python_modules"]:
        assert len(entry["source_blob"]) == 40
        assert (PROJECT_ROOT / entry["public_path"]).is_file()
    for entry in payload["formal_corpora"]:
        assert len(entry["source_tree"]) == 40
        assert (PROJECT_ROOT / entry["public_path"]).is_dir()
