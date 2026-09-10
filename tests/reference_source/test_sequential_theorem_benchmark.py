from __future__ import annotations

import pytest

from causal_observation_reproduction.reference.sequential_theorem_benchmark import (
    ALIGNMENT_PANEL,
    NULL_PANEL,
    PATHOLOGY_PANEL,
    POSITIVE_PANEL,
    run_panel,
)


def test_positive_panel_measures_future_only_strict_refinement_value() -> None:
    payload = run_panel(POSITIVE_PANEL, seeds=(1, 2, 3, 4), rollouts=1024)

    assert payload["immediate_reward_identical"] is True
    assert payload["future_transition_reward_only"] is True
    assert payload["sequential_value_gap_delta"] > 0.1
    assert all(world["positive_reachability"] for world in payload["worlds"])
    assert all(world["incompatible_optimal_actions"] for world in payload["worlds"])
    assert all(world["deterministic_refinement"] for world in payload["worlds"])


def test_null_panel_has_no_value_when_continuation_action_is_shared() -> None:
    payload = run_panel(NULL_PANEL, seeds=(1, 2, 3, 4), rollouts=512)

    assert payload["sequential_value_gap_delta_null"] == pytest.approx(0.0)
    assert all(not world["incompatible_optimal_actions"] for world in payload["worlds"])


def test_pathology_panel_reports_negative_net_value_when_assumptions_fail() -> None:
    payload = run_panel(PATHOLOGY_PANEL, seeds=(1, 2, 3, 4), rollouts=512)

    assert payload["sequential_value_gap_delta_pathology"] < 0.0
    assert all(not world["deterministic_refinement"] for world in payload["worlds"])


def test_alignment_panel_compares_bellman_prediction_with_empirical_return() -> None:
    payload = run_panel(ALIGNMENT_PANEL, seeds=tuple(range(1, 17)), rollouts=4096)

    value = payload["theorem_conditioned_value_gap_explained_variance"]
    assert 0.0 <= value <= 1.0
    assert value > 0.5
