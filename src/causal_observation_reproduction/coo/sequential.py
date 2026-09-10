"""Exact finite-horizon sequential causal-observation planning.

The one-step COO rule compares two observation maps once.  This module lifts
that rule to a finite-horizon belief-state problem in which an information
action is chosen before every control action.  A supplied decision-relative
quotient can be validated and compiled into a smaller, value-preserving model;
the same planner can then run on either the ambient or quotient model.

The implementation is intentionally finite and explicit.  Every reward,
transition kernel, observation kernel, sensing cost, and budget charge is part
of the public problem definition, which keeps Bellman decisions replayable and
lets experiments report declared evaluation work without using wall-clock time.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import TYPE_CHECKING, NoReturn, TypeAlias

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

StateId = str
ControlActionId = str
InformationActionId = str
ObservationId = str
QuotientClassId = str
Belief: TypeAlias = tuple[float, ...]
RewardKey: TypeAlias = tuple[int, StateId, ControlActionId]
TransitionKey: TypeAlias = tuple[int, StateId, ControlActionId]
ObservationKey: TypeAlias = tuple[int, InformationActionId, StateId]
ObservationProjectionKey: TypeAlias = tuple[int, InformationActionId, ObservationId]


def _raise_invalid(message: str) -> NoReturn:
    raise ValueError(message)


class InvalidSequentialQuotient(ValueError):
    """Raised when a proposed quotient is not value preserving."""


def _raise_invalid_quotient(message: str) -> NoReturn:
    raise InvalidSequentialQuotient(message)


def _raise_key_error(message: str) -> NoReturn:
    raise KeyError(message)


def _raise_internal_error(message: str) -> NoReturn:
    raise AssertionError(message)


@dataclass(frozen=True, slots=True)
class InformationActionSpec:
    """One sensing choice, its utility cost, and its integer budget charge."""

    name: InformationActionId
    cost: float = 0.0
    budget_units: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            _raise_invalid("information action names must be nonempty")
        if self.cost < 0.0:
            _raise_invalid("information action costs must be nonnegative")
        if self.budget_units < 0:
            _raise_invalid("information action budget units must be nonnegative")


@dataclass(frozen=True)
class FiniteSequentialCooProblem:
    """A finite partially observed control problem with explicit sensing.

    At each time step the planner chooses an information action, receives an
    observation from its declared kernel, chooses a control action, receives a
    reward, and transitions without observing the next latent state directly.
    """

    states: tuple[StateId, ...]
    control_actions: tuple[ControlActionId, ...]
    information_actions: tuple[InformationActionSpec, ...]
    horizon: int
    initial_probabilities: Mapping[StateId, float]
    rewards: Mapping[RewardKey, float]
    transitions: Mapping[TransitionKey, Mapping[StateId, float]]
    observations: Mapping[ObservationKey, Mapping[ObservationId, float]]

    def __post_init__(self) -> None:
        _validate_problem(self)

    @property
    def initial_belief(self) -> Belief:
        """Return the initial distribution in declared state order."""

        return tuple(float(self.initial_probabilities[state]) for state in self.states)

    def information_action(self, name: InformationActionId) -> InformationActionSpec:
        """Resolve an information action by its stable identifier."""

        for action in self.information_actions:
            if action.name == name:
                return action
        message = f"unknown information action: {name!r}"
        _raise_key_error(message)


@dataclass(frozen=True)
class SequentialQuotient:
    """A proposed decision-relative state and observation projection."""

    state_to_class: Mapping[StateId, QuotientClassId]
    observation_to_class: Mapping[ObservationProjectionKey, ObservationId]


@dataclass(frozen=True)
class QuotientProjection:
    """A validated reduced problem plus maps needed at deployment time."""

    problem: FiniteSequentialCooProblem
    state_to_class: Mapping[StateId, QuotientClassId]
    observation_to_class: Mapping[ObservationProjectionKey, ObservationId]

    def project_belief(
        self, ambient_problem: FiniteSequentialCooProblem, belief: Belief
    ) -> Belief:
        """Project an ambient belief onto quotient classes."""

        _validate_belief(ambient_problem, belief)
        probabilities = {
            state: belief[index] for index, state in enumerate(ambient_problem.states)
        }
        return tuple(
            sum(
                probabilities[state]
                for state in ambient_problem.states
                if self.state_to_class[state] == quotient_class
            )
            for quotient_class in self.problem.states
        )

    def project_observation(
        self,
        *,
        time: int,
        information_action: InformationActionId,
        observation: ObservationId,
    ) -> ObservationId:
        """Map one ambient observation to its quotient-visible class."""

        key = (time, information_action, observation)
        try:
            return self.observation_to_class[key]
        except KeyError as error:
            message = f"observation projection is undefined for {key!r}"
            raise KeyError(message) from error


@dataclass(frozen=True, slots=True)
class PlannerWork:
    """Declared Bellman evaluation work for one information decision."""

    information_action_candidates: int = 0
    observation_branches: int = 0
    control_action_candidates: int = 0
    latent_state_terms: int = 0

    def __add__(self, other: PlannerWork) -> PlannerWork:
        return PlannerWork(
            information_action_candidates=(
                self.information_action_candidates + other.information_action_candidates
            ),
            observation_branches=self.observation_branches + other.observation_branches,
            control_action_candidates=(
                self.control_action_candidates + other.control_action_candidates
            ),
            latent_state_terms=self.latent_state_terms + other.latent_state_terms,
        )

    @property
    def total(self) -> int:
        """Return a transparent scalar summary of declared work."""

        return (
            self.information_action_candidates
            + self.observation_branches
            + self.control_action_candidates
            + self.latent_state_terms
        )


@dataclass(frozen=True, slots=True)
class ObservationBranch:
    """One observation-contingent control decision in a meta-action value."""

    observation: ObservationId
    probability: float
    posterior: Belief
    control_action: ControlActionId
    continuation_value: float


@dataclass(frozen=True, slots=True)
class InformationActionValue:
    """Cost-adjusted Bellman value of one information action."""

    information_action: InformationActionId
    expected_value: float
    branches: tuple[ObservationBranch, ...]
    represented_probability: float
    work: PlannerWork

    def control_action_for(self, observation: ObservationId) -> ControlActionId | None:
        """Return the planned control action for a represented branch."""

        for branch in self.branches:
            if branch.observation == observation:
                return branch.control_action
        return None


@dataclass(frozen=True, slots=True)
class SequentialCooDecision:
    """The selected information action and all audited candidate values."""

    time: int
    budget_remaining: int
    belief: Belief
    selected_information_action: InformationActionId
    expected_value: float
    candidates: tuple[InformationActionValue, ...]
    work: PlannerWork

    @property
    def selected_candidate(self) -> InformationActionValue:
        """Return the candidate selected by the Bellman maximization."""

        for candidate in self.candidates:
            if candidate.information_action == self.selected_information_action:
                return candidate
        _raise_internal_error("selected information action is absent from candidates")


class SequentialCooPlanner:
    """Finite-horizon belief-state Bellman planner.

    ``max_observation_branches`` provides the matched bounded-VoI variant.  The
    most probable branches are evaluated and renormalized for candidate
    selection; if execution reaches an omitted branch, the policy uses the
    immediate posterior-optimal control action and resumes planning next step.
    """

    def __init__(
        self,
        problem: FiniteSequentialCooProblem,
        *,
        max_observation_branches: int | None = None,
    ) -> None:
        if max_observation_branches is not None and max_observation_branches < 1:
            _raise_invalid("max observation branches must be positive")
        self.problem = problem
        self.max_observation_branches = max_observation_branches
        self._value_cache: dict[tuple[int, int, Belief], float] = {}
        self._decision_cache: dict[tuple[int, int, Belief], SequentialCooDecision] = {}

    def decide(
        self, *, time: int, belief: Belief, budget_remaining: int
    ) -> SequentialCooDecision:
        """Choose an information action from the current sufficient statistic."""

        _validate_planning_state(self.problem, time, belief, budget_remaining)
        key = (time, budget_remaining, _belief_key(belief))
        cached = self._decision_cache.get(key)
        if cached is not None:
            return cached
        candidates = tuple(
            self._evaluate_information_action(
                time=time,
                belief=belief,
                budget_remaining=budget_remaining,
                information_action=action,
            )
            for action in self.problem.information_actions
            if action.budget_units <= budget_remaining
        )
        selected = max(candidates, key=lambda candidate: candidate.expected_value)
        work = PlannerWork()
        for candidate in candidates:
            work += candidate.work
        decision = SequentialCooDecision(
            time=time,
            budget_remaining=budget_remaining,
            belief=belief,
            selected_information_action=selected.information_action,
            expected_value=selected.expected_value,
            candidates=candidates,
            work=work,
        )
        self._decision_cache[key] = decision
        self._value_cache[key] = selected.expected_value
        return decision

    def posterior(
        self,
        *,
        time: int,
        belief: Belief,
        information_action: InformationActionId,
        observation: ObservationId,
    ) -> Belief:
        """Update the sufficient statistic after an observed sensing result."""

        return posterior_belief(
            self.problem,
            time=time,
            belief=belief,
            information_action=information_action,
            observation=observation,
        )

    def control_action(
        self,
        *,
        decision: SequentialCooDecision,
        observation: ObservationId,
    ) -> ControlActionId:
        """Return the observation-contingent control action for execution."""

        candidate = decision.selected_candidate
        planned = candidate.control_action_for(observation)
        if planned is not None:
            return planned
        posterior = self.posterior(
            time=decision.time,
            belief=decision.belief,
            information_action=candidate.information_action,
            observation=observation,
        )
        return _best_immediate_control(self.problem, decision.time, posterior)

    def predict(
        self,
        *,
        time: int,
        posterior: Belief,
        control_action: ControlActionId,
    ) -> Belief:
        """Predict the next latent-state belief after an unobserved transition."""

        return predict_belief(
            self.problem,
            time=time,
            belief=posterior,
            control_action=control_action,
        )

    def _evaluate_information_action(
        self,
        *,
        time: int,
        belief: Belief,
        budget_remaining: int,
        information_action: InformationActionSpec,
    ) -> InformationActionValue:
        distribution = observation_distribution(
            self.problem,
            time=time,
            belief=belief,
            information_action=information_action.name,
        )
        ordered = sorted(distribution.items(), key=lambda item: (-item[1], item[0]))
        if self.max_observation_branches is not None:
            ordered = ordered[: self.max_observation_branches]
        represented_probability = sum(probability for _, probability in ordered)
        if represented_probability <= 0.0:
            _raise_invalid("an information action has no represented observation mass")
        budget_after = budget_remaining - information_action.budget_units
        branches: list[ObservationBranch] = []
        expected_value = -information_action.cost
        for observation, probability in ordered:
            posterior = posterior_belief(
                self.problem,
                time=time,
                belief=belief,
                information_action=information_action.name,
                observation=observation,
            )
            control_action, continuation_value = self._best_control(
                time=time,
                posterior=posterior,
                budget_remaining=budget_after,
            )
            normalized_probability = probability / represented_probability
            expected_value += normalized_probability * continuation_value
            branches.append(
                ObservationBranch(
                    observation=observation,
                    probability=probability,
                    posterior=posterior,
                    control_action=control_action,
                    continuation_value=continuation_value,
                )
            )
        branch_count = len(branches)
        control_count = branch_count * len(self.problem.control_actions)
        state_terms = control_count * len(self.problem.states)
        return InformationActionValue(
            information_action=information_action.name,
            expected_value=expected_value,
            branches=tuple(branches),
            represented_probability=represented_probability,
            work=PlannerWork(
                information_action_candidates=1,
                observation_branches=branch_count,
                control_action_candidates=control_count,
                latent_state_terms=state_terms,
            ),
        )

    def _best_control(
        self, *, time: int, posterior: Belief, budget_remaining: int
    ) -> tuple[ControlActionId, float]:
        values = tuple(
            (
                action,
                expected_reward(self.problem, time, posterior, action)
                + self._future_value(
                    time=time,
                    posterior=posterior,
                    budget_remaining=budget_remaining,
                    control_action=action,
                ),
            )
            for action in self.problem.control_actions
        )
        return max(values, key=lambda item: item[1])

    def _future_value(
        self,
        *,
        time: int,
        posterior: Belief,
        budget_remaining: int,
        control_action: ControlActionId,
    ) -> float:
        if time + 1 >= self.problem.horizon:
            return 0.0
        next_belief = predict_belief(
            self.problem,
            time=time,
            belief=posterior,
            control_action=control_action,
        )
        return self._value(
            time=time + 1,
            belief=next_belief,
            budget_remaining=budget_remaining,
        )

    def _value(self, *, time: int, belief: Belief, budget_remaining: int) -> float:
        key = (time, budget_remaining, _belief_key(belief))
        cached = self._value_cache.get(key)
        if cached is not None:
            return cached
        return self.decide(
            time=time,
            belief=belief,
            budget_remaining=budget_remaining,
        ).expected_value


def compile_sequential_quotient(
    problem: FiniteSequentialCooProblem,
    quotient: SequentialQuotient,
) -> QuotientProjection:
    """Validate and compile a decision-relative quotient.

    Rewards must agree inside every class, controlled transitions must be
    lumpable, and projected observation kernels must agree inside each class.
    These finite checks are sufficient for the reduced Bellman recursion to
    preserve the ambient model's value for policies expressed on the quotient.
    """

    if set(quotient.state_to_class) != set(problem.states):
        _raise_invalid_quotient(
            "state quotient must be defined for exactly the ambient states"
        )
    if any(not value for value in quotient.state_to_class.values()):
        _raise_invalid_quotient("quotient class identifiers must be nonempty")
    classes = tuple(
        dict.fromkeys(quotient.state_to_class[state] for state in problem.states)
    )
    members = {
        quotient_class: tuple(
            state
            for state in problem.states
            if quotient.state_to_class[state] == quotient_class
        )
        for quotient_class in classes
    }
    _validate_observation_projection(problem, quotient.observation_to_class)
    rewards: dict[RewardKey, float] = {}
    transitions: dict[TransitionKey, dict[StateId, float]] = {}
    observations: dict[ObservationKey, dict[ObservationId, float]] = {}
    for time in range(problem.horizon):
        for quotient_class in classes:
            class_members = members[quotient_class]
            representative = class_members[0]
            for action in problem.control_actions:
                reference_reward = float(
                    problem.rewards[(time, representative, action)]
                )
                if any(
                    not isclose(
                        float(problem.rewards[(time, state, action)]),
                        reference_reward,
                        abs_tol=1e-12,
                    )
                    for state in class_members[1:]
                ):
                    _raise_invalid_quotient(
                        "reward differs inside a proposed quotient class"
                    )
                rewards[(time, quotient_class, action)] = reference_reward
                if time + 1 < problem.horizon:
                    reference_transition = _project_transition_distribution(
                        problem,
                        time=time,
                        state=representative,
                        control_action=action,
                        state_to_class=quotient.state_to_class,
                        classes=classes,
                    )
                    for state in class_members[1:]:
                        candidate = _project_transition_distribution(
                            problem,
                            time=time,
                            state=state,
                            control_action=action,
                            state_to_class=quotient.state_to_class,
                            classes=classes,
                        )
                        if not _distributions_close(reference_transition, candidate):
                            _raise_invalid_quotient(
                                "transition kernel is not lumpable inside a quotient class"
                            )
                    transitions[(time, quotient_class, action)] = reference_transition
            for information_action in problem.information_actions:
                reference_observation = _project_observation_distribution(
                    problem,
                    time=time,
                    state=representative,
                    information_action=information_action.name,
                    observation_projection=quotient.observation_to_class,
                )
                for state in class_members[1:]:
                    candidate = _project_observation_distribution(
                        problem,
                        time=time,
                        state=state,
                        information_action=information_action.name,
                        observation_projection=quotient.observation_to_class,
                    )
                    if not _distributions_close(reference_observation, candidate):
                        _raise_invalid_quotient(
                            "projected observation kernel differs inside a quotient class"
                        )
                observations[(time, information_action.name, quotient_class)] = (
                    reference_observation
                )
    initial_probabilities = {
        quotient_class: sum(
            float(problem.initial_probabilities[state])
            for state in members[quotient_class]
        )
        for quotient_class in classes
    }
    reduced = FiniteSequentialCooProblem(
        states=classes,
        control_actions=problem.control_actions,
        information_actions=problem.information_actions,
        horizon=problem.horizon,
        initial_probabilities=initial_probabilities,
        rewards=rewards,
        transitions=transitions,
        observations=observations,
    )
    return QuotientProjection(
        problem=reduced,
        state_to_class=dict(quotient.state_to_class),
        observation_to_class=dict(quotient.observation_to_class),
    )


def observation_distribution(
    problem: FiniteSequentialCooProblem,
    *,
    time: int,
    belief: Belief,
    information_action: InformationActionId,
) -> dict[ObservationId, float]:
    """Return the predictive observation distribution for one meta action."""

    _validate_belief(problem, belief)
    distribution: dict[ObservationId, float] = {}
    for index, state in enumerate(problem.states):
        for observation, probability in problem.observations[
            (time, information_action, state)
        ].items():
            distribution[observation] = distribution.get(observation, 0.0) + (
                belief[index] * float(probability)
            )
    return {
        observation: probability
        for observation, probability in sorted(distribution.items())
        if probability > 1e-15
    }


def posterior_belief(
    problem: FiniteSequentialCooProblem,
    *,
    time: int,
    belief: Belief,
    information_action: InformationActionId,
    observation: ObservationId,
) -> Belief:
    """Apply Bayes' rule for one information-action observation."""

    _validate_belief(problem, belief)
    weights = tuple(
        belief[index]
        * float(
            problem.observations[(time, information_action, state)].get(
                observation, 0.0
            )
        )
        for index, state in enumerate(problem.states)
    )
    total = sum(weights)
    if total <= 0.0:
        _raise_invalid("cannot condition on a zero-probability observation")
    return tuple(weight / total for weight in weights)


def predict_belief(
    problem: FiniteSequentialCooProblem,
    *,
    time: int,
    belief: Belief,
    control_action: ControlActionId,
) -> Belief:
    """Push a posterior through the controlled transition kernel."""

    _validate_belief(problem, belief)
    if time + 1 >= problem.horizon:
        return belief
    probabilities = dict.fromkeys(problem.states, 0.0)
    for index, state in enumerate(problem.states):
        for next_state, probability in problem.transitions[
            (time, state, control_action)
        ].items():
            probabilities[next_state] += belief[index] * float(probability)
    return tuple(probabilities[state] for state in problem.states)


def expected_reward(
    problem: FiniteSequentialCooProblem,
    time: int,
    belief: Belief,
    control_action: ControlActionId,
) -> float:
    """Return posterior expected immediate control reward."""

    return sum(
        belief[index] * float(problem.rewards[(time, state, control_action)])
        for index, state in enumerate(problem.states)
    )


def _best_immediate_control(
    problem: FiniteSequentialCooProblem, time: int, belief: Belief
) -> ControlActionId:
    values = tuple(
        (action, expected_reward(problem, time, belief, action))
        for action in problem.control_actions
    )
    return max(values, key=lambda item: item[1])[0]


def _validate_problem(problem: FiniteSequentialCooProblem) -> None:
    if problem.horizon < 1:
        _raise_invalid("sequential horizon must be positive")
    if not problem.states or len(set(problem.states)) != len(problem.states):
        _raise_invalid("states must be nonempty and unique")
    if not problem.control_actions or len(set(problem.control_actions)) != len(
        problem.control_actions
    ):
        _raise_invalid("control actions must be nonempty and unique")
    information_names = tuple(action.name for action in problem.information_actions)
    if not information_names or len(set(information_names)) != len(information_names):
        _raise_invalid("information actions must be nonempty and unique")
    if set(problem.initial_probabilities) != set(problem.states):
        _raise_invalid("initial probabilities must cover exactly the states")
    _validate_distribution(problem.initial_probabilities, "initial probabilities")
    expected_rewards = {
        (time, state, action)
        for time in range(problem.horizon)
        for state in problem.states
        for action in problem.control_actions
    }
    if set(problem.rewards) != expected_rewards:
        _raise_invalid("rewards must cover every time-state-control triple")
    expected_transitions = {
        (time, state, action)
        for time in range(problem.horizon - 1)
        for state in problem.states
        for action in problem.control_actions
    }
    if set(problem.transitions) != expected_transitions:
        _raise_invalid("transitions must cover every nonterminal state-control triple")
    for distribution in problem.transitions.values():
        if set(distribution) != set(problem.states):
            _raise_invalid("transition distributions must cover exactly the states")
        _validate_distribution(distribution, "transition distribution")
    expected_observations = {
        (time, information_action, state)
        for time in range(problem.horizon)
        for information_action in information_names
        for state in problem.states
    }
    if set(problem.observations) != expected_observations:
        _raise_invalid("observation kernels must cover every time-action-state triple")
    for distribution in problem.observations.values():
        if not distribution or any(not observation for observation in distribution):
            _raise_invalid("observation distributions need nonempty identifiers")
        _validate_distribution(distribution, "observation distribution")


def _validate_distribution(distribution: Mapping[str, float], label: str) -> None:
    if any(float(probability) < 0.0 for probability in distribution.values()):
        _raise_invalid(f"{label} must be nonnegative")
    if not isclose(
        sum(float(value) for value in distribution.values()), 1.0, abs_tol=1e-12
    ):
        _raise_invalid(f"{label} must sum to one")


def _validate_belief(problem: FiniteSequentialCooProblem, belief: Belief) -> None:
    if len(belief) != len(problem.states):
        _raise_invalid("belief dimension must match the state space")
    if any(probability < -1e-12 for probability in belief):
        _raise_invalid("belief probabilities must be nonnegative")
    if not isclose(sum(belief), 1.0, abs_tol=1e-9):
        _raise_invalid("belief probabilities must sum to one")


def _validate_planning_state(
    problem: FiniteSequentialCooProblem,
    time: int,
    belief: Belief,
    budget_remaining: int,
) -> None:
    if not 0 <= time < problem.horizon:
        _raise_invalid("planning time must be inside the finite horizon")
    if budget_remaining < 0:
        _raise_invalid("remaining sensing budget must be nonnegative")
    _validate_belief(problem, belief)


def _belief_key(belief: Belief) -> Belief:
    normalized = tuple(
        0.0 if abs(value) < 1e-14 else round(value, 14) for value in belief
    )
    total = sum(normalized)
    if total == 0.0:
        _raise_invalid("belief key cannot have zero total mass")
    return tuple(value / total for value in normalized)


def _validate_observation_projection(
    problem: FiniteSequentialCooProblem,
    projection: Mapping[ObservationProjectionKey, ObservationId],
) -> None:
    expected = {
        (time, information_action.name, observation)
        for time in range(problem.horizon)
        for information_action in problem.information_actions
        for state in problem.states
        for observation in problem.observations[(time, information_action.name, state)]
    }
    if set(projection) != expected:
        _raise_invalid_quotient(
            "observation projection must cover every reachable ambient observation"
        )
    if any(not value for value in projection.values()):
        _raise_invalid_quotient("projected observation identifiers must be nonempty")


def _project_transition_distribution(
    problem: FiniteSequentialCooProblem,
    *,
    time: int,
    state: StateId,
    control_action: ControlActionId,
    state_to_class: Mapping[StateId, QuotientClassId],
    classes: Sequence[QuotientClassId],
) -> dict[StateId, float]:
    distribution = problem.transitions[(time, state, control_action)]
    return {
        quotient_class: sum(
            float(probability)
            for next_state, probability in distribution.items()
            if state_to_class[next_state] == quotient_class
        )
        for quotient_class in classes
    }


def _project_observation_distribution(
    problem: FiniteSequentialCooProblem,
    *,
    time: int,
    state: StateId,
    information_action: InformationActionId,
    observation_projection: Mapping[ObservationProjectionKey, ObservationId],
) -> dict[ObservationId, float]:
    projected: dict[ObservationId, float] = {}
    for observation, probability in problem.observations[
        (time, information_action, state)
    ].items():
        quotient_observation = observation_projection[
            (time, information_action, observation)
        ]
        projected[quotient_observation] = projected.get(
            quotient_observation, 0.0
        ) + float(probability)
    return dict(sorted(projected.items()))


def _distributions_close(left: Mapping[str, float], right: Mapping[str, float]) -> bool:
    keys = set(left) | set(right)
    return all(
        isclose(float(left.get(key, 0.0)), float(right.get(key, 0.0)), abs_tol=1e-12)
        for key in keys
    )


__all__ = [
    "Belief",
    "ControlActionId",
    "FiniteSequentialCooProblem",
    "InformationActionId",
    "InformationActionSpec",
    "InformationActionValue",
    "InvalidSequentialQuotient",
    "ObservationBranch",
    "ObservationId",
    "PlannerWork",
    "QuotientProjection",
    "SequentialCooDecision",
    "SequentialCooPlanner",
    "SequentialQuotient",
    "StateId",
    "compile_sequential_quotient",
    "expected_reward",
    "observation_distribution",
    "posterior_belief",
    "predict_belief",
]
