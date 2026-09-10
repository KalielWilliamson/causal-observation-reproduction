"""Independent-sample robustness study for history-dependent sequential COO.

This experiment is deliberately separate from the frozen paper-confirmation
protocol.  It exercises the general belief-state planner over multiple sensing
decisions, budgets, causal design families, and paired random worlds.  Raw
episode rows remain the canonical evidence; aggregates and bootstrap intervals
are derived views marked pending independent review.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isclose
from pathlib import Path
from statistics import fmean
from typing import TYPE_CHECKING, Literal, NoReturn, TypeAlias

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

from causal_observation_reproduction.coo.sequential import (
    Belief,
    FiniteSequentialCooProblem,
    InformationActionSpec,
    InvalidSequentialQuotient,
    PlannerWork,
    QuotientProjection,
    SequentialCooPlanner,
    SequentialQuotient,
    compile_sequential_quotient,
    expected_reward,
    posterior_belief,
    predict_belief,
)

SCHEMA_VERSION = "causal-observation-reproduction.sequential-robustness.v1"
EVIDENCE_STATUS = "robustness_extension_pending_independent_review"

Regime = Literal["positive", "null", "pathology", "invalid"]
PolicyName = Literal[
    "sequential_coo",
    "generic_bounded_voi",
    "generic_exact_voi",
    "never_acquire",
    "always_causal_probe",
]
TransitionMode = Literal["static", "cycle", "resample", "action_driven"]
CellKey: TypeAlias = tuple[str, Regime, int, int, float]

SEQUENTIAL_COO: PolicyName = "sequential_coo"
GENERIC_BOUNDED: PolicyName = "generic_bounded_voi"
GENERIC_EXACT: PolicyName = "generic_exact_voi"
NEVER_ACQUIRE: PolicyName = "never_acquire"
ALWAYS_CAUSAL: PolicyName = "always_causal_probe"
POLICIES: tuple[PolicyName, ...] = (
    SEQUENTIAL_COO,
    GENERIC_BOUNDED,
    GENERIC_EXACT,
    NEVER_ACQUIRE,
    ALWAYS_CAUSAL,
)


def _raise_invalid(message: str) -> NoReturn:
    raise ValueError(message)


@dataclass(frozen=True, slots=True)
class SequentialDesignFamily:
    """One outcome-blind causal design family used by the robustness matrix."""

    name: str
    nuisance_cardinality: int
    early_coarse_reliability: float
    late_coarse_reliability: float
    alias_persistence: int
    causal_probe_reliability: float
    nuisance_probe_reliability: float
    target_nuisance_correlation: float
    transition_mode: TransitionMode

    def __post_init__(self) -> None:
        if not self.name:
            _raise_invalid("design-family name must be nonempty")
        if self.nuisance_cardinality < 2:
            _raise_invalid("nuisance cardinality must be at least two")
        for value in (
            self.early_coarse_reliability,
            self.late_coarse_reliability,
            self.causal_probe_reliability,
            self.nuisance_probe_reliability,
        ):
            if not 0.5 <= value <= 1.0:
                _raise_invalid("observation reliability must be in [0.5, 1]")
        if self.alias_persistence < 0:
            _raise_invalid("alias persistence must be nonnegative")
        if not -0.95 <= self.target_nuisance_correlation <= 0.95:
            _raise_invalid("target-nuisance correlation must be in [-0.95, 0.95]")


DESIGN_FAMILIES: tuple[SequentialDesignFamily, ...] = (
    SequentialDesignFamily("binary_fork", 2, 0.55, 0.90, 2, 0.90, 1.00, 0.00, "static"),
    SequentialDesignFamily("wide_fork", 8, 0.55, 0.85, 2, 0.90, 0.95, 0.00, "static"),
    SequentialDesignFamily(
        "mediated_cycle", 4, 0.60, 0.85, 3, 0.82, 0.90, 0.25, "cycle"
    ),
    SequentialDesignFamily(
        "resampled_nuisance", 4, 0.60, 0.80, 2, 0.86, 0.85, -0.20, "resample"
    ),
    SequentialDesignFamily(
        "action_driven_nuisance", 4, 0.58, 0.82, 3, 0.84, 0.90, 0.30, "action_driven"
    ),
    SequentialDesignFamily(
        "persistent_alias", 8, 0.52, 0.65, 6, 0.92, 0.95, 0.10, "cycle"
    ),
    SequentialDesignFamily(
        "noisy_boundary", 4, 0.62, 0.78, 2, 0.70, 0.75, -0.30, "resample"
    ),
    SequentialDesignFamily(
        "redundant_boundary", 2, 0.80, 0.96, 1, 0.80, 1.00, 0.00, "action_driven"
    ),
)


@dataclass(frozen=True, slots=True)
class SequentialRobustnessConfig:
    """Independent-sample factorial design for the sequential extension."""

    design_families: tuple[str, ...] = tuple(family.name for family in DESIGN_FAMILIES)
    regimes: tuple[Regime, ...] = ("positive", "null", "pathology", "invalid")
    horizons: tuple[int, ...] = (4, 6)
    sensing_budgets: tuple[int, ...] = (1, 3)
    acquisition_costs: tuple[float, ...] = (0.05, 0.20)
    evaluation_seeds: tuple[int, ...] = tuple(range(501, 629))
    generic_branch_budget: int = 4
    bootstrap_samples: int = 2_000
    noninferiority_margin: float = 0.05

    def __post_init__(self) -> None:
        known = {family.name for family in DESIGN_FAMILIES}
        if not self.design_families or not set(self.design_families) <= known:
            _raise_invalid("robustness design families must be known and nonempty")
        if not self.regimes or not self.horizons or not self.sensing_budgets:
            _raise_invalid("robustness factorial axes must be nonempty")
        if any(horizon < 2 for horizon in self.horizons):
            _raise_invalid("sequential robustness horizons must be at least two")
        if any(budget < 0 for budget in self.sensing_budgets):
            _raise_invalid("sensing budgets must be nonnegative")
        if any(cost < 0.0 for cost in self.acquisition_costs):
            _raise_invalid("acquisition costs must be nonnegative")
        if not self.evaluation_seeds or len(set(self.evaluation_seeds)) != len(
            self.evaluation_seeds
        ):
            _raise_invalid("evaluation seeds must be nonempty and unique")
        if self.generic_branch_budget < 1 or self.bootstrap_samples < 1:
            _raise_invalid("branch and bootstrap sample counts must be positive")
        if self.noninferiority_margin < 0.0:
            _raise_invalid("noninferiority margin must be nonnegative")


@dataclass(frozen=True, slots=True)
class SequentialProblemBundle:
    """Ambient problem and its proposed causal/query quotient."""

    problem: FiniteSequentialCooProblem
    quotient: SequentialQuotient
    quotient_expected_valid: bool


@dataclass(frozen=True, slots=True)
class SequentialRobustnessEpisode:
    """Replayable paired episode evidence."""

    cell_id: str
    design_family: str
    regime: Regime
    horizon: int
    sensing_budget: int
    acquisition_cost: float
    seed: int
    world_id: str
    policy: PolicyName
    quotient_status: str
    initial_expected_value: float | None
    cost_adjusted_return: float
    control_reward: float
    sensing_cost: float
    sensing_actions: tuple[str, ...]
    control_actions: tuple[str, ...]
    information_action_candidates: int
    observation_branches: int
    control_action_candidates: int
    latent_state_terms: int
    declared_work: int


@dataclass(frozen=True, slots=True)
class SequentialRobustnessAggregate:
    """Mean metrics for one design cell and policy."""

    cell_id: str
    design_family: str
    regime: Regime
    horizon: int
    sensing_budget: int
    acquisition_cost: float
    policy: PolicyName
    episode_count: int
    mean_cost_adjusted_return: float
    mean_control_reward: float
    mean_sensing_cost: float
    mean_probe_count: float
    mean_declared_work: float


@dataclass(frozen=True, slots=True)
class PairedRobustnessComparison:
    """Paired COO comparison against one generic VoI reference."""

    cell_id: str
    design_family: str
    regime: Regime
    horizon: int
    sensing_budget: int
    acquisition_cost: float
    reference_policy: PolicyName
    pair_count: int
    mean_return_difference: float
    return_difference_ci_lower: float
    return_difference_ci_upper: float
    initial_expected_value_difference: float
    initial_expected_values_equal: bool
    mean_work_ratio: float
    work_ratio_ci_lower: float
    work_ratio_ci_upper: float
    return_noninferior: bool
    work_reduced: bool


@dataclass(frozen=True, slots=True)
class SequentialRobustnessResult:
    """In-memory result and written artifact paths."""

    episodes: tuple[SequentialRobustnessEpisode, ...]
    aggregates: tuple[SequentialRobustnessAggregate, ...]
    comparisons: tuple[PairedRobustnessComparison, ...]
    artifacts: tuple[Path, ...]


def build_sequential_problem(
    family: SequentialDesignFamily,
    regime: Regime,
    *,
    horizon: int,
    acquisition_cost: float,
) -> SequentialProblemBundle:
    """Build one finite POSCM-like problem and its declared query quotient."""

    states = tuple(
        f"target-{target}:nuisance-{nuisance}"
        for target in (0, 1)
        for nuisance in range(family.nuisance_cardinality)
    )
    control_actions = ("0", "1")
    probe_reliability = (
        min(family.causal_probe_reliability, 0.62)
        if regime == "pathology"
        else family.causal_probe_reliability
    )
    probe_cost = (
        max(acquisition_cost, 0.25) if regime == "pathology" else acquisition_cost
    )
    information_actions = (
        InformationActionSpec("coarse"),
        InformationActionSpec("causal_probe", cost=probe_cost, budget_units=1),
        InformationActionSpec("nuisance_probe", cost=probe_cost * 0.75, budget_units=1),
    )
    initial_probabilities = _initial_probabilities(states, family)
    rewards = _reward_table(
        states=states,
        actions=control_actions,
        horizon=horizon,
        regime=regime,
    )
    transitions = _transition_table(
        states=states,
        actions=control_actions,
        horizon=horizon,
        family=family,
    )
    observations = _observation_table(
        states=states,
        horizon=horizon,
        family=family,
        causal_probe_reliability=probe_reliability,
    )
    problem = FiniteSequentialCooProblem(
        states=states,
        control_actions=control_actions,
        information_actions=information_actions,
        horizon=horizon,
        initial_probabilities=initial_probabilities,
        rewards=rewards,
        transitions=transitions,
        observations=observations,
    )
    quotient = _query_quotient(problem, regime=regime)
    return SequentialProblemBundle(
        problem=problem,
        quotient=quotient,
        quotient_expected_valid=regime != "invalid",
    )


def run_sequential_robustness(
    config: SequentialRobustnessConfig | None = None,
) -> SequentialRobustnessResult:
    """Execute the independent paired matrix without writing artifacts."""

    resolved = config or SequentialRobustnessConfig()
    family_index = {family.name: family for family in DESIGN_FAMILIES}
    episodes: list[SequentialRobustnessEpisode] = []
    for family_name in resolved.design_families:
        family = family_index[family_name]
        for regime in resolved.regimes:
            for horizon in resolved.horizons:
                for acquisition_cost in resolved.acquisition_costs:
                    bundle = build_sequential_problem(
                        family,
                        regime,
                        horizon=horizon,
                        acquisition_cost=acquisition_cost,
                    )
                    planners = _policy_planners(bundle, resolved)
                    for sensing_budget in resolved.sensing_budgets:
                        cell_id = _cell_id(
                            family.name,
                            regime,
                            horizon,
                            sensing_budget,
                            acquisition_cost,
                        )
                        for seed in resolved.evaluation_seeds:
                            for policy in POLICIES:
                                planner, projection, quotient_status = planners[policy]
                                episodes.append(
                                    _run_episode(
                                        bundle.problem,
                                        policy=policy,
                                        planner=planner,
                                        projection=projection,
                                        quotient_status=quotient_status,
                                        cell_id=cell_id,
                                        family=family,
                                        regime=regime,
                                        sensing_budget=sensing_budget,
                                        acquisition_cost=acquisition_cost,
                                        seed=seed,
                                    )
                                )
    episode_tuple = tuple(episodes)
    aggregates = _aggregate_episodes(episode_tuple)
    comparisons = _paired_comparisons(episode_tuple, resolved)
    return SequentialRobustnessResult(
        episodes=episode_tuple,
        aggregates=aggregates,
        comparisons=comparisons,
        artifacts=(),
    )


def write_sequential_robustness(
    output_dir: str | Path,
    config: SequentialRobustnessConfig | None = None,
) -> tuple[Path, Path, Path, Path]:
    """Write JSONL evidence, derived summaries, and a design manifest."""

    resolved = config or SequentialRobustnessConfig()
    result = run_sequential_robustness(resolved)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    episodes_path = directory / "sequential_robustness_episodes.jsonl"
    aggregates_path = directory / "sequential_robustness_aggregates.jsonl"
    comparisons_path = directory / "sequential_robustness_comparisons.jsonl"
    manifest_path = directory / "sequential_robustness_manifest.json"
    _write_jsonl(episodes_path, result.episodes)
    _write_jsonl(aggregates_path, result.aggregates)
    _write_jsonl(comparisons_path, result.comparisons)
    valid_comparisons = tuple(
        row
        for row in result.comparisons
        if row.regime != "invalid" and row.reference_policy == GENERIC_EXACT
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "evidence_status": EVIDENCE_STATUS,
        "historical_protocol_modified": False,
        "config": asdict(resolved),
        "design_families": [
            asdict(family)
            for family in DESIGN_FAMILIES
            if family.name in resolved.design_families
        ],
        "episode_count": len(result.episodes),
        "aggregate_count": len(result.aggregates),
        "paired_comparison_count": len(result.comparisons),
        "independent_world_count": len({row.world_id for row in result.episodes}),
        "claim_checks": {
            "valid_quotient_return_noninferiority_to_exact_count": sum(
                int(row.return_noninferior) for row in valid_comparisons
            ),
            "valid_quotient_comparison_count": len(valid_comparisons),
            "valid_quotient_work_reduction_count": sum(
                int(row.work_reduced) for row in valid_comparisons
            ),
            "valid_quotient_bellman_value_equality_count": sum(
                int(row.initial_expected_values_equal) for row in valid_comparisons
            ),
            "paper_level_conclusion": False,
        },
        "artifacts": {
            "episodes": episodes_path.name,
            "aggregates": aggregates_path.name,
            "comparisons": comparisons_path.name,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return episodes_path, aggregates_path, comparisons_path, manifest_path


def _policy_planners(
    bundle: SequentialProblemBundle,
    config: SequentialRobustnessConfig,
) -> dict[
    PolicyName,
    tuple[SequentialCooPlanner | None, QuotientProjection | None, str],
]:
    ambient_exact = SequentialCooPlanner(bundle.problem)
    ambient_bounded = SequentialCooPlanner(
        bundle.problem, max_observation_branches=config.generic_branch_budget
    )
    try:
        projection = compile_sequential_quotient(bundle.problem, bundle.quotient)
    except InvalidSequentialQuotient:
        if bundle.quotient_expected_valid:
            raise
        quotient_planner = ambient_exact
        quotient_projection = None
        quotient_status = "invalid_quotient_safe_full_model_fallback"
    else:
        if not bundle.quotient_expected_valid:
            _raise_invalid("an invalid-control quotient unexpectedly passed validation")
        quotient_planner = SequentialCooPlanner(projection.problem)
        quotient_projection = projection
        quotient_status = "validated_value_preserving_quotient"
    return {
        SEQUENTIAL_COO: (
            quotient_planner,
            quotient_projection,
            quotient_status,
        ),
        GENERIC_BOUNDED: (ambient_bounded, None, "ambient_branch_bounded_model"),
        GENERIC_EXACT: (ambient_exact, None, "ambient_exact_model"),
        NEVER_ACQUIRE: (None, None, "fixed_control"),
        ALWAYS_CAUSAL: (None, None, "fixed_control"),
    }


def _run_episode(
    problem: FiniteSequentialCooProblem,
    *,
    policy: PolicyName,
    planner: SequentialCooPlanner | None,
    projection: QuotientProjection | None,
    quotient_status: str,
    cell_id: str,
    family: SequentialDesignFamily,
    regime: Regime,
    sensing_budget: int,
    acquisition_cost: float,
    seed: int,
) -> SequentialRobustnessEpisode:
    rng = random.Random(_stable_seed(cell_id, seed))
    state = _sample_distribution(problem.initial_probabilities, rng)
    ambient_belief = problem.initial_belief
    planning_belief = (
        projection.project_belief(problem, ambient_belief)
        if projection is not None
        else ambient_belief
    )
    budget_remaining = sensing_budget
    total_control_reward = 0.0
    total_sensing_cost = 0.0
    sensing_actions: list[str] = []
    control_actions: list[str] = []
    total_work = PlannerWork()
    initial_expected_value: float | None = None
    for time in range(problem.horizon):
        decision = None
        if planner is not None:
            decision = planner.decide(
                time=time,
                belief=planning_belief,
                budget_remaining=budget_remaining,
            )
            information_action = decision.selected_information_action
            total_work += decision.work
            if time == 0:
                initial_expected_value = decision.expected_value
        elif policy == ALWAYS_CAUSAL and budget_remaining > 0:
            information_action = "causal_probe"
        else:
            information_action = "coarse"
        action_spec = problem.information_action(information_action)
        observation = _sample_distribution(
            problem.observations[(time, information_action, state)], rng
        )
        ambient_posterior = posterior_belief(
            problem,
            time=time,
            belief=ambient_belief,
            information_action=information_action,
            observation=observation,
        )
        planning_observation = (
            projection.project_observation(
                time=time,
                information_action=information_action,
                observation=observation,
            )
            if projection is not None
            else observation
        )
        planning_posterior = (
            planner.posterior(
                time=time,
                belief=planning_belief,
                information_action=information_action,
                observation=planning_observation,
            )
            if planner is not None
            else ambient_posterior
        )
        if planner is not None and decision is not None:
            control_action = planner.control_action(
                decision=decision, observation=planning_observation
            )
        else:
            control_action = _best_immediate_control(problem, time, ambient_posterior)
        control_reward = float(problem.rewards[(time, state, control_action)])
        total_control_reward += control_reward
        total_sensing_cost += action_spec.cost
        budget_remaining -= action_spec.budget_units
        sensing_actions.append(information_action)
        control_actions.append(control_action)
        if time + 1 < problem.horizon:
            state = _sample_distribution(
                problem.transitions[(time, state, control_action)], rng
            )
            ambient_belief = predict_belief(
                problem,
                time=time,
                belief=ambient_posterior,
                control_action=control_action,
            )
            if planner is not None:
                planning_belief = planner.predict(
                    time=time,
                    posterior=planning_posterior,
                    control_action=control_action,
                )
            else:
                planning_belief = ambient_belief
    return SequentialRobustnessEpisode(
        cell_id=cell_id,
        design_family=family.name,
        regime=regime,
        horizon=problem.horizon,
        sensing_budget=sensing_budget,
        acquisition_cost=acquisition_cost,
        seed=seed,
        world_id=_world_id(cell_id, seed),
        policy=policy,
        quotient_status=quotient_status,
        initial_expected_value=initial_expected_value,
        cost_adjusted_return=total_control_reward - total_sensing_cost,
        control_reward=total_control_reward,
        sensing_cost=total_sensing_cost,
        sensing_actions=tuple(sensing_actions),
        control_actions=tuple(control_actions),
        information_action_candidates=total_work.information_action_candidates,
        observation_branches=total_work.observation_branches,
        control_action_candidates=total_work.control_action_candidates,
        latent_state_terms=total_work.latent_state_terms,
        declared_work=total_work.total,
    )


def _initial_probabilities(
    states: Sequence[str], family: SequentialDesignFamily
) -> dict[str, float]:
    weights: dict[str, float] = {}
    for state in states:
        target, nuisance = _state_values(state)
        parity_matches = nuisance % 2 == target
        correlation = family.target_nuisance_correlation
        weights[state] = 1.0 + (correlation if parity_matches else -correlation)
    total = sum(weights.values())
    return {state: weight / total for state, weight in weights.items()}


def _reward_table(
    *,
    states: Sequence[str],
    actions: Sequence[str],
    horizon: int,
    regime: Regime,
) -> dict[tuple[int, str, str], float]:
    rewards: dict[tuple[int, str, str], float] = {}
    decision_relevance_time = max(1, horizon // 3)
    for time in range(horizon):
        for state in states:
            target, nuisance = _state_values(state)
            desired = (
                0
                if regime == "null"
                else target ^ (nuisance % 2)
                if regime == "invalid"
                else target
            )
            for action in actions:
                rewards[(time, state, action)] = (
                    0.0
                    if time < decision_relevance_time
                    else float(int(action) == desired)
                )
    return rewards


def _transition_table(
    *,
    states: Sequence[str],
    actions: Sequence[str],
    horizon: int,
    family: SequentialDesignFamily,
) -> dict[tuple[int, str, str], dict[str, float]]:
    table: dict[tuple[int, str, str], dict[str, float]] = {}
    for time in range(horizon - 1):
        for state in states:
            target, nuisance = _state_values(state)
            for action in actions:
                distribution = dict.fromkeys(states, 0.0)
                if family.transition_mode == "resample":
                    for next_nuisance in range(family.nuisance_cardinality):
                        distribution[_state_id(target, next_nuisance)] = (
                            1.0 / family.nuisance_cardinality
                        )
                else:
                    next_nuisance = nuisance
                    if family.transition_mode == "cycle" or (
                        family.transition_mode == "action_driven" and action == "1"
                    ):
                        next_nuisance = (nuisance + 1) % family.nuisance_cardinality
                    distribution[_state_id(target, next_nuisance)] = 1.0
                table[(time, state, action)] = distribution
    return table


def _observation_table(
    *,
    states: Sequence[str],
    horizon: int,
    family: SequentialDesignFamily,
    causal_probe_reliability: float,
) -> dict[tuple[int, str, str], dict[str, float]]:
    table: dict[tuple[int, str, str], dict[str, float]] = {}
    for time in range(horizon):
        coarse_reliability = (
            family.early_coarse_reliability
            if time < family.alias_persistence
            else family.late_coarse_reliability
        )
        for state in states:
            target, nuisance = _state_values(state)
            coarse = _binary_distribution(target, coarse_reliability)
            table[(time, "coarse", state)] = {
                f"coarse-{value}": probability for value, probability in coarse.items()
            }
            causal = _binary_distribution(target, causal_probe_reliability)
            table[(time, "causal_probe", state)] = _product_observation(
                coarse, causal, left="coarse", right="causal"
            )
            nuisance_distribution = _categorical_distribution(
                nuisance,
                family.nuisance_cardinality,
                family.nuisance_probe_reliability,
            )
            table[(time, "nuisance_probe", state)] = _product_observation(
                coarse,
                nuisance_distribution,
                left="coarse",
                right="nuisance",
            )
    return table


def _query_quotient(
    problem: FiniteSequentialCooProblem, *, regime: Regime
) -> SequentialQuotient:
    if regime == "null":
        state_to_class = dict.fromkeys(problem.states, "constant-action")
    else:
        state_to_class = {
            state: f"target-{_state_values(state)[0]}" for state in problem.states
        }
    observation_to_class: dict[tuple[int, str, str], str] = {}
    for time in range(problem.horizon):
        for information_action in problem.information_actions:
            observations = {
                observation
                for state in problem.states
                for observation in problem.observations[
                    (time, information_action.name, state)
                ]
            }
            for observation in observations:
                if regime == "null":
                    projected = "constant-query-observation"
                elif information_action.name == "nuisance_probe":
                    projected = observation.split(";", maxsplit=1)[0] + ";nuisance-*"
                else:
                    projected = observation
                observation_to_class[(time, information_action.name, observation)] = (
                    projected
                )
    return SequentialQuotient(
        state_to_class=state_to_class,
        observation_to_class=observation_to_class,
    )


def _binary_distribution(value: int, reliability: float) -> dict[int, float]:
    return {value: reliability, 1 - value: 1.0 - reliability}


def _categorical_distribution(
    value: int, cardinality: int, reliability: float
) -> dict[int, float]:
    remaining = (1.0 - reliability) / (cardinality - 1)
    return {
        candidate: reliability if candidate == value else remaining
        for candidate in range(cardinality)
    }


def _product_observation(
    left_distribution: Mapping[int, float],
    right_distribution: Mapping[int, float],
    *,
    left: str,
    right: str,
) -> dict[str, float]:
    return {
        f"{left}-{left_value};{right}-{right_value}": left_probability
        * right_probability
        for left_value, left_probability in left_distribution.items()
        for right_value, right_probability in right_distribution.items()
    }


def _aggregate_episodes(
    episodes: Sequence[SequentialRobustnessEpisode],
) -> tuple[SequentialRobustnessAggregate, ...]:
    grouped: dict[tuple[CellKey, PolicyName], list[SequentialRobustnessEpisode]] = (
        defaultdict(list)
    )
    for row in episodes:
        grouped[(_episode_cell_key(row), row.policy)].append(row)
    aggregates: list[SequentialRobustnessAggregate] = []
    for (_, policy), rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        first = rows[0]
        aggregates.append(
            SequentialRobustnessAggregate(
                cell_id=first.cell_id,
                design_family=first.design_family,
                regime=first.regime,
                horizon=first.horizon,
                sensing_budget=first.sensing_budget,
                acquisition_cost=first.acquisition_cost,
                policy=policy,
                episode_count=len(rows),
                mean_cost_adjusted_return=fmean(
                    row.cost_adjusted_return for row in rows
                ),
                mean_control_reward=fmean(row.control_reward for row in rows),
                mean_sensing_cost=fmean(row.sensing_cost for row in rows),
                mean_probe_count=fmean(
                    sum(action != "coarse" for action in row.sensing_actions)
                    for row in rows
                ),
                mean_declared_work=fmean(row.declared_work for row in rows),
            )
        )
    return tuple(aggregates)


def _paired_comparisons(
    episodes: Sequence[SequentialRobustnessEpisode],
    config: SequentialRobustnessConfig,
) -> tuple[PairedRobustnessComparison, ...]:
    indexed = {(_episode_cell_key(row), row.seed, row.policy): row for row in episodes}
    cells = sorted({_episode_cell_key(row) for row in episodes}, key=str)
    comparisons: list[PairedRobustnessComparison] = []
    for cell in cells:
        family_name, regime, horizon, sensing_budget, acquisition_cost = cell
        cell_id = _cell_id(
            family_name, regime, horizon, sensing_budget, acquisition_cost
        )
        for reference in (GENERIC_BOUNDED, GENERIC_EXACT):
            differences: list[float] = []
            initial_value_differences: list[float] = []
            work_ratios: list[float] = []
            for seed in config.evaluation_seeds:
                coo = indexed[(cell, seed, SEQUENTIAL_COO)]
                generic = indexed[(cell, seed, reference)]
                differences.append(
                    coo.cost_adjusted_return - generic.cost_adjusted_return
                )
                if (
                    coo.initial_expected_value is None
                    or generic.initial_expected_value is None
                ):
                    _raise_invalid("planner comparisons need initial Bellman values")
                initial_value_differences.append(
                    coo.initial_expected_value - generic.initial_expected_value
                )
                work_ratios.append(
                    coo.declared_work / generic.declared_work
                    if generic.declared_work > 0
                    else 1.0
                )
            difference_interval = _paired_bootstrap_interval(
                differences,
                samples=config.bootstrap_samples,
                seed=_stable_seed(cell_id + reference, 91),
            )
            work_interval = _paired_bootstrap_interval(
                work_ratios,
                samples=config.bootstrap_samples,
                seed=_stable_seed(cell_id + reference, 193),
            )
            comparisons.append(
                PairedRobustnessComparison(
                    cell_id=cell_id,
                    design_family=family_name,
                    regime=regime,
                    horizon=horizon,
                    sensing_budget=sensing_budget,
                    acquisition_cost=acquisition_cost,
                    reference_policy=reference,
                    pair_count=len(differences),
                    mean_return_difference=fmean(differences),
                    return_difference_ci_lower=difference_interval[0],
                    return_difference_ci_upper=difference_interval[1],
                    initial_expected_value_difference=fmean(initial_value_differences),
                    initial_expected_values_equal=all(
                        isclose(value, 0.0, abs_tol=1e-12)
                        for value in initial_value_differences
                    ),
                    mean_work_ratio=fmean(work_ratios),
                    work_ratio_ci_lower=work_interval[0],
                    work_ratio_ci_upper=work_interval[1],
                    return_noninferior=(
                        difference_interval[0] >= -config.noninferiority_margin
                    ),
                    work_reduced=work_interval[1] < 1.0,
                )
            )
    return tuple(comparisons)


def _paired_bootstrap_interval(
    values: Sequence[float], *, samples: int, seed: int
) -> tuple[float, float]:
    if not values:
        _raise_invalid("bootstrap input must be nonempty")
    if all(isclose(values[0], value, abs_tol=1e-15) for value in values[1:]):
        return values[0], values[0]
    rng = random.Random(seed)
    means = sorted(
        fmean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(samples)
    )
    lower = means[int(0.025 * (samples - 1))]
    upper = means[int(0.975 * (samples - 1))]
    return lower, upper


def _best_immediate_control(
    problem: FiniteSequentialCooProblem, time: int, belief: Belief
) -> str:
    values = tuple(
        (action, expected_reward(problem, time, belief, action))
        for action in problem.control_actions
    )
    return max(values, key=lambda item: item[1])[0]


def _sample_distribution(distribution: Mapping[str, float], rng: random.Random) -> str:
    threshold = rng.random()
    cumulative = 0.0
    last = ""
    for value, probability in distribution.items():
        last = value
        cumulative += float(probability)
        if threshold <= cumulative:
            return value
    if not last:
        _raise_invalid("cannot sample an empty distribution")
    return last


def _state_values(state: str) -> tuple[int, int]:
    target_text, nuisance_text = state.split(":", maxsplit=1)
    return int(target_text.removeprefix("target-")), int(
        nuisance_text.removeprefix("nuisance-")
    )


def _state_id(target: int, nuisance: int) -> str:
    return f"target-{target}:nuisance-{nuisance}"


def _cell_id(
    family: str, regime: Regime, horizon: int, sensing_budget: int, cost: float
) -> str:
    return f"{family}:{regime}:h{horizon}:b{sensing_budget}:c{cost:.2f}"


def _episode_cell_key(row: SequentialRobustnessEpisode) -> CellKey:
    return (
        row.design_family,
        row.regime,
        row.horizon,
        row.sensing_budget,
        row.acquisition_cost,
    )


def _stable_seed(text: str, seed: int) -> int:
    digest = sha256(f"{text}:{seed}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _world_id(cell_id: str, seed: int) -> str:
    digest = sha256(f"{cell_id}:{seed}".encode()).hexdigest()[:16]
    return f"sequential-robustness-{digest}"


def _write_jsonl(
    path: Path,
    rows: Iterable[
        SequentialRobustnessEpisode
        | SequentialRobustnessAggregate
        | PairedRobustnessComparison
    ],
) -> None:
    path.write_text(
        "".join(json.dumps(asdict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


__all__ = [
    "DESIGN_FAMILIES",
    "EVIDENCE_STATUS",
    "PairedRobustnessComparison",
    "SequentialDesignFamily",
    "SequentialProblemBundle",
    "SequentialRobustnessAggregate",
    "SequentialRobustnessConfig",
    "SequentialRobustnessEpisode",
    "SequentialRobustnessResult",
    "build_sequential_problem",
    "run_sequential_robustness",
    "write_sequential_robustness",
]
