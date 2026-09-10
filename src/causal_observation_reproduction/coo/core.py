"""Finite decision-relative observation quotient calculations.

This module is the algorithmic core shared by the paper experiments.  It
evaluates an observation map by pulling the optimal decision rule back onto
its equivalence classes, then decides whether a refined map is worth its
declared acquisition cost.  It deliberately knows nothing about experiment
tracking, environments, or output formats.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import TYPE_CHECKING, NoReturn, TypeAlias

if TYPE_CHECKING:
    from collections.abc import Mapping


ContextId = str
ActionId = str
ObservationId = str
UtilityKey = tuple[ContextId, ActionId]
ObservationKernel: TypeAlias = "Mapping[ContextId, Mapping[ObservationId, float]]"


def _raise_invalid(message: str) -> NoReturn:
    """Raise a uniform input-contract error from validation helpers."""

    raise ValueError(message)


@dataclass(frozen=True)
class FiniteDecisionProblem:
    """A finite decision problem over latent contexts and control actions."""

    contexts: tuple[ContextId, ...]
    actions: tuple[ActionId, ...]
    probabilities: Mapping[ContextId, float]
    utilities: Mapping[UtilityKey, float]

    def __post_init__(self) -> None:
        if not self.contexts:
            _raise_invalid("a decision problem needs at least one context")
        if not self.actions:
            _raise_invalid("a decision problem needs at least one action")
        if len(set(self.contexts)) != len(self.contexts):
            _raise_invalid("context identifiers must be unique")
        if len(set(self.actions)) != len(self.actions):
            _raise_invalid("action identifiers must be unique")
        probability_keys = set(self.probabilities)
        expected_contexts = set(self.contexts)
        if probability_keys != expected_contexts:
            _raise_invalid("probabilities must be defined for exactly the contexts")
        if any(value < 0.0 for value in self.probabilities.values()):
            _raise_invalid("context probabilities must be nonnegative")
        if not isclose(sum(self.probabilities.values()), 1.0, abs_tol=1e-12):
            _raise_invalid("context probabilities must sum to one")
        expected_utilities = {
            (context, action) for context in self.contexts for action in self.actions
        }
        if set(self.utilities) != expected_utilities:
            _raise_invalid("utilities must be defined for every context-action pair")


@dataclass(frozen=True)
class ObservationEvaluation:
    """Optimal expected value and induced action rule for an observation map."""

    value: float
    policy: tuple[tuple[ObservationId, ActionId], ...]

    def action_for(self, observation: ObservationId) -> ActionId:
        """Return the stable optimal action assigned to an observation class."""

        for observation_id, action in self.policy:
            if observation_id == observation:
                return action
        message = f"unknown observation class: {observation!r}"
        raise KeyError(message)


@dataclass(frozen=True)
class AcquisitionDecision:
    """Cost-adjusted evaluation of replacing a coarse map with a refinement."""

    coarse: ObservationEvaluation
    refined: ObservationEvaluation
    acquisition_cost: float
    gross_value: float
    net_value: float
    acquire: bool


def evaluate_observation(
    problem: FiniteDecisionProblem,
    observation_map: Mapping[ContextId, ObservationId],
) -> ObservationEvaluation:
    """Optimize the control action independently within each observation class.

    Ties resolve by the declared action order, making outputs deterministic and
    preserving an explicit policy even when information has zero value.
    """

    _validate_observation_map(problem, observation_map)
    kernel: ObservationKernel = {
        context: {observation_map[context]: 1.0} for context in problem.contexts
    }
    return evaluate_kernel(problem, kernel)


def evaluate_kernel(
    problem: FiniteDecisionProblem,
    observation_kernel: ObservationKernel,
) -> ObservationEvaluation:
    """Optimize actions under a finite, possibly noisy observation kernel."""

    _validate_observation_kernel(problem, observation_kernel)
    classes = tuple(
        sorted(
            {
                observation
                for distribution in observation_kernel.values()
                for observation in distribution
            }
        )
    )
    policy_rows: list[tuple[ObservationId, ActionId]] = []
    value = 0.0
    for observation in classes:
        action_values = tuple(
            (
                action,
                sum(
                    problem.probabilities[context]
                    * observation_kernel[context].get(observation, 0.0)
                    * problem.utilities[(context, action)]
                    for context in problem.contexts
                ),
            )
            for action in problem.actions
        )
        action, class_value = max(action_values, key=lambda item: item[1])
        policy_rows.append((observation, action))
        value += class_value
    return ObservationEvaluation(value=value, policy=tuple(policy_rows))


def decide_refinement(
    problem: FiniteDecisionProblem,
    *,
    coarse_map: Mapping[ContextId, ObservationId],
    refined_map: Mapping[ContextId, ObservationId],
    acquisition_cost: float,
) -> AcquisitionDecision:
    """Evaluate the COO acquisition rule for a declared map refinement.

    A valid refined map may split a coarse observation class but may not merge
    two distinct coarse classes.  The returned decision is the paper's
    cost-adjusted criterion: acquire exactly when refined value minus coarse
    value and declared cost is strictly positive.
    """

    if acquisition_cost < 0.0:
        _raise_invalid("acquisition cost must be nonnegative")
    _validate_observation_map(problem, coarse_map)
    _validate_observation_map(problem, refined_map)
    _validate_refinement(problem, coarse_map=coarse_map, refined_map=refined_map)
    coarse = evaluate_observation(problem, coarse_map)
    refined = evaluate_observation(problem, refined_map)
    gross_value = refined.value - coarse.value
    net_value = gross_value - acquisition_cost
    return AcquisitionDecision(
        coarse=coarse,
        refined=refined,
        acquisition_cost=acquisition_cost,
        gross_value=gross_value,
        net_value=net_value,
        acquire=net_value > 0.0,
    )


def decide_kernel_refinement(
    problem: FiniteDecisionProblem,
    *,
    coarse_kernel: ObservationKernel,
    refined_kernel: ObservationKernel,
    acquisition_cost: float,
) -> AcquisitionDecision:
    """Apply the cost-adjusted COO rule to declared noisy observation kernels."""

    if acquisition_cost < 0.0:
        _raise_invalid("acquisition cost must be nonnegative")
    coarse = evaluate_kernel(problem, coarse_kernel)
    refined = evaluate_kernel(problem, refined_kernel)
    gross_value = refined.value - coarse.value
    net_value = gross_value - acquisition_cost
    return AcquisitionDecision(
        coarse=coarse,
        refined=refined,
        acquisition_cost=acquisition_cost,
        gross_value=gross_value,
        net_value=net_value,
        acquire=net_value > 0.0,
    )


def _validate_observation_map(
    problem: FiniteDecisionProblem,
    observation_map: Mapping[ContextId, ObservationId],
) -> None:
    if set(observation_map) != set(problem.contexts):
        _raise_invalid("observation map must be defined for exactly the contexts")
    if any(not observation for observation in observation_map.values()):
        _raise_invalid("observation identifiers must be nonempty")


def _validate_observation_kernel(
    problem: FiniteDecisionProblem, observation_kernel: ObservationKernel
) -> None:
    if set(observation_kernel) != set(problem.contexts):
        _raise_invalid("observation kernel must be defined for exactly the contexts")
    for distribution in observation_kernel.values():
        if not distribution:
            _raise_invalid("every context needs a nonempty observation distribution")
        if any(not observation for observation in distribution):
            _raise_invalid("observation identifiers must be nonempty")
        if any(value < 0.0 for value in distribution.values()):
            _raise_invalid("observation probabilities must be nonnegative")
        if not isclose(sum(distribution.values()), 1.0, abs_tol=1e-12):
            _raise_invalid("observation probabilities must sum to one")


def _validate_refinement(
    problem: FiniteDecisionProblem,
    *,
    coarse_map: Mapping[ContextId, ObservationId],
    refined_map: Mapping[ContextId, ObservationId],
) -> None:
    for left_index, left in enumerate(problem.contexts):
        for right in problem.contexts[left_index + 1 :]:
            if (
                refined_map[left] == refined_map[right]
                and coarse_map[left] != coarse_map[right]
            ):
                _raise_invalid("a refinement may not merge distinct coarse classes")


__all__ = [
    "AcquisitionDecision",
    "ActionId",
    "ContextId",
    "FiniteDecisionProblem",
    "ObservationEvaluation",
    "ObservationId",
    "ObservationKernel",
    "decide_kernel_refinement",
    "decide_refinement",
    "evaluate_kernel",
    "evaluate_observation",
]
