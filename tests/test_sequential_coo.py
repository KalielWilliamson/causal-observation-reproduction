"""Tests for the history-dependent sequential COO Bellman planner."""

from __future__ import annotations

from dataclasses import replace

import pytest

from causal_observation_reproduction.coo.sequential import (
    FiniteSequentialCooProblem,
    InformationActionSpec,
    InvalidSequentialQuotient,
    SequentialCooPlanner,
    SequentialQuotient,
    compile_sequential_quotient,
)


def _binary_target_problem(*, horizon: int = 2) -> FiniteSequentialCooProblem:
    states = (
        "target-0:nuisance-0",
        "target-0:nuisance-1",
        "target-1:nuisance-0",
        "target-1:nuisance-1",
    )
    control_actions = ("0", "1")
    information_actions = (
        InformationActionSpec("none"),
        InformationActionSpec("causal_probe", cost=0.1, budget_units=1),
        InformationActionSpec("nuisance_probe", cost=0.1, budget_units=1),
    )
    rewards = {
        (time, state, action): (
            0.0 if time == 0 else float(action == state.split(":", maxsplit=1)[0][-1])
        )
        for time in range(horizon)
        for state in states
        for action in control_actions
    }
    transitions = {
        (time, state, action): {
            candidate: float(candidate == state) for candidate in states
        }
        for time in range(horizon - 1)
        for state in states
        for action in control_actions
    }
    observations: dict[tuple[int, str, str], dict[str, float]] = {}
    for time in range(horizon):
        for state in states:
            target = state.split(":", maxsplit=1)[0][-1]
            nuisance = state[-1]
            observations[(time, "none", state)] = {"coarse": 1.0}
            observations[(time, "causal_probe", state)] = (
                {f"target-{target}": 1.0} if time == 0 else {"expired": 1.0}
            )
            observations[(time, "nuisance_probe", state)] = {
                f"nuisance-{nuisance}": 1.0
            }
    return FiniteSequentialCooProblem(
        states=states,
        control_actions=control_actions,
        information_actions=information_actions,
        horizon=horizon,
        initial_probabilities=dict.fromkeys(states, 0.25),
        rewards=rewards,
        transitions=transitions,
        observations=observations,
    )


def _target_quotient(problem: FiniteSequentialCooProblem) -> SequentialQuotient:
    state_to_class = {
        state: state.split(":", maxsplit=1)[0] for state in problem.states
    }
    observation_to_class: dict[tuple[int, str, str], str] = {}
    for time in range(problem.horizon):
        observation_to_class[(time, "none", "coarse")] = "coarse"
        if time == 0:
            for target in ("0", "1"):
                observation_to_class[(time, "causal_probe", f"target-{target}")] = (
                    f"target-{target}"
                )
        else:
            observation_to_class[(time, "causal_probe", "expired")] = "expired"
        for nuisance in ("0", "1"):
            observation_to_class[(time, "nuisance_probe", f"nuisance-{nuisance}")] = (
                "nuisance-collapsed"
            )
    return SequentialQuotient(
        state_to_class=state_to_class,
        observation_to_class=observation_to_class,
    )


def test_sequential_coo_uses_bellman_continuation_value() -> None:
    problem = _binary_target_problem()
    planner = SequentialCooPlanner(problem)

    decision = planner.decide(time=0, belief=problem.initial_belief, budget_remaining=1)

    assert decision.selected_information_action == "causal_probe"
    assert decision.expected_value == pytest.approx(0.9)
    assert {candidate.information_action for candidate in decision.candidates} == {
        "none",
        "causal_probe",
        "nuisance_probe",
    }


def test_information_choice_changes_with_the_belief_history() -> None:
    problem = _binary_target_problem(horizon=3)
    planner = SequentialCooPlanner(problem)
    uncertain = planner.decide(
        time=0, belief=problem.initial_belief, budget_remaining=2
    )
    known_target = (0.5, 0.5, 0.0, 0.0)
    informed = planner.decide(time=0, belief=known_target, budget_remaining=2)

    assert uncertain.selected_information_action == "causal_probe"
    assert informed.selected_information_action == "none"


def test_valid_quotient_preserves_value_and_reduces_declared_work() -> None:
    problem = _binary_target_problem()
    projection = compile_sequential_quotient(problem, _target_quotient(problem))
    ambient = SequentialCooPlanner(problem).decide(
        time=0, belief=problem.initial_belief, budget_remaining=1
    )
    quotient = SequentialCooPlanner(projection.problem).decide(
        time=0,
        belief=projection.project_belief(problem, problem.initial_belief),
        budget_remaining=1,
    )

    assert quotient.expected_value == pytest.approx(ambient.expected_value)
    assert quotient.selected_information_action == ambient.selected_information_action
    assert quotient.work.total < ambient.work.total
    assert quotient.work.observation_branches < ambient.work.observation_branches


def test_invalid_quotient_is_rejected_when_rewards_vary_inside_a_class() -> None:
    problem = _binary_target_problem()
    rewards = dict(problem.rewards)
    rewards[(1, "target-0:nuisance-1", "0")] = 0.25
    invalid_problem = replace(problem, rewards=rewards)

    with pytest.raises(InvalidSequentialQuotient, match="reward differs"):
        compile_sequential_quotient(invalid_problem, _target_quotient(invalid_problem))


def test_bounded_generic_planner_declares_truncated_probability_mass() -> None:
    problem = _binary_target_problem()
    planner = SequentialCooPlanner(problem, max_observation_branches=1)

    decision = planner.decide(time=0, belief=problem.initial_belief, budget_remaining=1)
    causal = next(
        candidate
        for candidate in decision.candidates
        if candidate.information_action == "causal_probe"
    )

    assert causal.represented_probability == pytest.approx(0.5)
    assert len(causal.branches) == 1
