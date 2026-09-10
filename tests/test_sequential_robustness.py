"""Acceptance tests for the independent sequential COO robustness study."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from causal_observation_reproduction.coo.sequential import (
    InvalidSequentialQuotient,
    SequentialCooPlanner,
    compile_sequential_quotient,
)
from causal_observation_reproduction.experiments.sequential_robustness import (
    DESIGN_FAMILIES,
    SequentialRobustnessConfig,
    build_sequential_problem,
    run_sequential_robustness,
    write_sequential_robustness,
)

if TYPE_CHECKING:
    from pathlib import Path


def _small_config() -> SequentialRobustnessConfig:
    return SequentialRobustnessConfig(
        design_families=("binary_fork", "noisy_boundary"),
        regimes=("positive", "null", "pathology", "invalid"),
        horizons=(3,),
        sensing_budgets=(2,),
        acquisition_costs=(0.05,),
        evaluation_seeds=(31, 32, 33, 34),
        generic_branch_budget=4,
        bootstrap_samples=100,
    )


def test_default_matrix_uses_broad_independent_design_families_and_seeds() -> None:
    config = SequentialRobustnessConfig()

    assert len(DESIGN_FAMILIES) == 8
    assert len(config.evaluation_seeds) == 128
    assert len(set(config.evaluation_seeds)) == 128
    assert config.sensing_budgets == (1, 3)


def test_valid_query_quotient_matches_ambient_bellman_value() -> None:
    family = next(family for family in DESIGN_FAMILIES if family.name == "wide_fork")
    bundle = build_sequential_problem(
        family, "positive", horizon=4, acquisition_cost=0.05
    )
    projection = compile_sequential_quotient(bundle.problem, bundle.quotient)
    ambient = SequentialCooPlanner(bundle.problem).decide(
        time=0, belief=bundle.problem.initial_belief, budget_remaining=2
    )
    quotient = SequentialCooPlanner(projection.problem).decide(
        time=0,
        belief=projection.project_belief(bundle.problem, bundle.problem.initial_belief),
        budget_remaining=2,
    )

    assert quotient.expected_value == pytest.approx(ambient.expected_value)
    assert quotient.selected_information_action == ambient.selected_information_action
    assert quotient.work.total < ambient.work.total


def test_misspecified_query_quotient_fails_closed() -> None:
    family = DESIGN_FAMILIES[0]
    bundle = build_sequential_problem(
        family, "invalid", horizon=3, acquisition_cost=0.05
    )

    with pytest.raises(InvalidSequentialQuotient, match="reward differs"):
        compile_sequential_quotient(bundle.problem, bundle.quotient)


def test_history_dependent_policy_can_probe_more_than_once() -> None:
    config = SequentialRobustnessConfig(
        design_families=("noisy_boundary",),
        regimes=("positive",),
        horizons=(6,),
        sensing_budgets=(3,),
        acquisition_costs=(0.05,),
        evaluation_seeds=tuple(range(1, 17)),
        bootstrap_samples=50,
    )

    result = run_sequential_robustness(config)
    coo_rows = [row for row in result.episodes if row.policy == "sequential_coo"]

    assert any(row.sensing_actions.count("causal_probe") > 1 for row in coo_rows)
    assert all(len(row.sensing_actions) == 6 for row in coo_rows)
    assert all(row.declared_work > 0 for row in coo_rows)


def test_paired_matrix_preserves_world_identity_and_invalid_fallback() -> None:
    result = run_sequential_robustness(_small_config())
    grouped: dict[tuple[str, int], set[str]] = {}
    for row in result.episodes:
        grouped.setdefault((row.cell_id, row.seed), set()).add(row.world_id)

    assert all(len(world_ids) == 1 for world_ids in grouped.values())
    invalid_coo = [
        row
        for row in result.episodes
        if row.regime == "invalid" and row.policy == "sequential_coo"
    ]
    assert invalid_coo
    assert {row.quotient_status for row in invalid_coo} == {
        "invalid_quotient_safe_full_model_fallback"
    }
    valid_exact = [
        row
        for row in result.comparisons
        if row.regime != "invalid" and row.reference_policy == "generic_exact_voi"
    ]
    assert valid_exact
    assert all(row.initial_expected_values_equal for row in valid_exact)


def test_writer_emits_raw_jsonl_and_review_pending_manifest(tmp_path: Path) -> None:
    paths = write_sequential_robustness(tmp_path, _small_config())
    manifest = json.loads(paths[-1].read_text(encoding="utf-8"))

    assert len(paths) == 4
    assert manifest["historical_protocol_modified"] is False
    assert manifest["evidence_status"].endswith("pending_independent_review")
    assert manifest["episode_count"] == 160
    assert manifest["independent_world_count"] == 32
    assert manifest["claim_checks"]["paper_level_conclusion"] is False
