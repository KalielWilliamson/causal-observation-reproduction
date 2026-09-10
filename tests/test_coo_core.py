from __future__ import annotations

import pytest

from causal_observation_reproduction.coo import (
    FiniteDecisionProblem,
    decide_kernel_refinement,
    decide_refinement,
    evaluate_observation,
)


def _binary_problem() -> FiniteDecisionProblem:
    return FiniteDecisionProblem(
        contexts=("zero", "one"),
        actions=("left", "right"),
        probabilities={"zero": 0.5, "one": 0.5},
        utilities={
            ("zero", "left"): 1.0,
            ("zero", "right"): 0.0,
            ("one", "left"): 0.0,
            ("one", "right"): 1.0,
        },
    )


def test_coo_acquires_only_when_refined_value_exceeds_cost() -> None:
    decision = decide_refinement(
        _binary_problem(),
        coarse_map={"zero": "coarse", "one": "coarse"},
        refined_map={"zero": "zero", "one": "one"},
        acquisition_cost=0.3,
    )

    assert decision.coarse.value == 0.5
    assert decision.refined.value == 1.0
    assert decision.gross_value == 0.5
    assert decision.net_value == 0.2
    assert decision.acquire is True


def test_coo_abstains_on_a_null_decision_difference() -> None:
    problem = FiniteDecisionProblem(
        contexts=("zero", "one"),
        actions=("left", "right"),
        probabilities={"zero": 0.5, "one": 0.5},
        utilities={
            ("zero", "left"): 1.0,
            ("zero", "right"): 0.0,
            ("one", "left"): 1.0,
            ("one", "right"): 0.0,
        },
    )

    decision = decide_refinement(
        problem,
        coarse_map={"zero": "coarse", "one": "coarse"},
        refined_map={"zero": "zero", "one": "one"},
        acquisition_cost=0.0,
    )

    assert decision.gross_value == 0.0
    assert decision.acquire is False


def test_refinement_cannot_merge_coarse_observation_classes() -> None:
    with pytest.raises(ValueError, match="may not merge"):
        decide_refinement(
            _binary_problem(),
            coarse_map={"zero": "zero", "one": "one"},
            refined_map={"zero": "refined", "one": "refined"},
            acquisition_cost=0.0,
        )


def test_observation_ties_use_declared_action_order() -> None:
    evaluation = evaluate_observation(
        _binary_problem(), {"zero": "coarse", "one": "coarse"}
    )

    assert evaluation.policy == (("coarse", "left"),)


def test_noisy_kernel_has_the_expected_refinement_value() -> None:
    decision = decide_kernel_refinement(
        _binary_problem(),
        coarse_kernel={"zero": {"coarse": 1.0}, "one": {"coarse": 1.0}},
        refined_kernel={
            "zero": {"zero": 0.8, "one": 0.2},
            "one": {"zero": 0.2, "one": 0.8},
        },
        acquisition_cost=0.15,
    )

    assert decision.coarse.value == 0.5
    assert decision.refined.value == 0.8
    assert decision.net_value == pytest.approx(0.15)
    assert decision.acquire is True
