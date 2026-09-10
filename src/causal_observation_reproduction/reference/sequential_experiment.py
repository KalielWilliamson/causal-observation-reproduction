"""Audited sequential confirmation for conditional causal-quotient sensing.

All information policies run through one paired-world episode interface. The
experiment separates a learned causal-quotient gate, bounded rollout-VoI,
same-capacity no-quotient control, simple public-feature baselines, and an
evaluator-only oracle. Training labels are paired returns; scorer sidecars are
rejected at the controller boundary and topology-instance clusters are the
unit of confirmation inference.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from statistics import fmean
from typing import Any, Literal, Protocol

from causal_observation_reproduction.reference.quotient_family import (
    CausalQuotientFamilySpec,
    GeneratedCausalQuotientFamily,
    generate_causal_quotient_family,
)
from causal_observation_reproduction.reference.quotient_recovery.statistics import (
    clustered_noninferiority,
)
from causal_observation_reproduction.reference.sequential_contract import (
    RUNNER_POLICY_ARMS,
)
from causal_observation_reproduction.reference.sequential_sensing import (
    SequentialQuotientSensingEnv,
)

SCHEMA_VERSION = (
    "causal_observation_reproduction.causal_quotient_conditional_efficiency_pilot.v1"
)
CONFIRMATION_SCHEMA_VERSION = (
    "causal_observation_reproduction.causal_quotient_confirmation_matrix.v1"
)
CLAIM_ID = "causal_quotient_conditional_voi_efficiency"
PolicyName = Literal[
    "learned_causal_quotient",
    "generic_bounded_rollout_voi",
    "capacity_matched_no_quotient",
    "never_refine",
    "always_refine",
    "random_budget_matched",
    "uncertainty",
    "information_gain",
    "oracle",
]
Regime = Literal["positive", "null", "invalid"]
POLICIES: tuple[PolicyName, ...] = (
    "never_refine",
    "always_refine",
    "random_budget_matched",
    "uncertainty",
    "information_gain",
    "generic_bounded_rollout_voi",
    "capacity_matched_no_quotient",
    "learned_causal_quotient",
    "oracle",
)
if tuple(POLICIES) != RUNNER_POLICY_ARMS:
    raise RuntimeError(
        "runner policy arms must be derived from the sequential contract"
    )


@dataclass(frozen=True)
class PolicyDecision:
    """One audited information-action decision made at deployment time."""

    acquire: bool
    rollout_count: int = 0
    branch_count: int = 0
    cache_hits: int = 0
    uses_oracle: bool = False


class InformationPolicy(Protocol):
    """Adapter boundary shared by every information-acquisition arm."""

    @property
    def name(self) -> PolicyName: ...

    def decide(
        self,
        *,
        features: Mapping[str, Any],
        seed: int,
        rollout: Callable[..., dict[str, Any]],
        probe_already_acquired: bool,
    ) -> PolicyDecision: ...

    def metadata(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class ConditionalEfficiencyPilotConfig:
    training_seeds: tuple[int, ...] = tuple(range(101, 117))
    evaluation_seeds: tuple[int, ...] = tuple(range(401, 417))
    rollout_budget: int = 16
    output_dir: str = "artifacts/causal_quotient_conditional_efficiency_pilot"

    def __post_init__(self) -> None:
        if self.rollout_budget < 1:
            raise ValueError("rollout_budget must be positive")


@dataclass(frozen=True)
class GateModel:
    """A threshold fitted from paired empirical return labels.

    ``uses_quotient`` selects the *only* feature difference between the two
    learned policies.  The feature is derived from the declared reward-query
    structure, not an oracle quotient label.
    """

    positive_gain: float
    nonpositive_gain: float
    uses_quotient: bool
    gain_by_feature_cost: tuple[tuple[bool, float, float], ...] = ()

    def acquire(self, *, query_disagreement: bool, probe_cost: float) -> bool:
        """Decide from the declared quotient feature and public protocol only."""
        signal = query_disagreement if self.uses_quotient else False
        cost = float(probe_cost)
        for feature, fitted_cost, gain in self.gain_by_feature_cost:
            if feature == signal and fitted_cost == cost:
                return gain > 0.0
        return (self.positive_gain if signal else self.nonpositive_gain) > 0.0

    def metadata(self) -> dict[str, Any]:
        return {
            "kind": "cost_conditioned_threshold_gate",
            "uses_query_feature": self.uses_quotient,
            "parameter_count": 2 + len(self.gain_by_feature_cost),
            "gain_by_feature_cost": [
                list(value) for value in self.gain_by_feature_cost
            ],
        }


@dataclass(frozen=True)
class GatePolicyAdapter:
    name: Literal["learned_causal_quotient", "capacity_matched_no_quotient"]
    model: GateModel

    def decide(
        self,
        *,
        features: Mapping[str, Any],
        seed: int,
        rollout: Callable[..., dict[str, Any]],
        probe_already_acquired: bool,
    ) -> PolicyDecision:
        _ = seed, rollout
        return PolicyDecision(
            acquire=not probe_already_acquired
            and self.model.acquire(
                query_disagreement=bool(features["query_disagreement"]),
                probe_cost=float(features["probe_cost"]),
            ),
        )

    def metadata(self) -> Mapping[str, Any]:
        return {
            "policy": self.name,
            "uses_oracle": False,
            "cache_policy": "disabled",
            "model": self.model.metadata(),
        }


@dataclass(frozen=True)
class SimplePolicyAdapter:
    name: Literal[
        "never_refine",
        "always_refine",
        "random_budget_matched",
        "uncertainty",
        "information_gain",
    ]
    random_activation_rate: float = 0.0

    def decide(
        self,
        *,
        features: Mapping[str, Any],
        seed: int,
        rollout: Callable[..., dict[str, Any]],
        probe_already_acquired: bool,
    ) -> PolicyDecision:
        _ = rollout
        if probe_already_acquired or not bool(features["probe_available"]):
            return PolicyDecision(acquire=False)
        if self.name == "never_refine":
            return PolicyDecision(acquire=False)
        if self.name == "always_refine":
            return PolicyDecision(acquire=True)
        if self.name == "random_budget_matched":
            # Deterministic, seed-indexed Bernoulli so paired rows are replayable.
            return PolicyDecision(
                acquire=((int(seed) * 1103515245 + 12345) % 2**31) / 2**31
                < self.random_activation_rate
            )
        if self.name == "uncertainty":
            return PolicyDecision(acquire=float(features["observation_noise"]) >= 0.25)
        if self.name == "information_gain":
            expected_gain = float(features["probe_reliability"]) * (
                1.0 - float(features["observation_noise"])
            )
            return PolicyDecision(acquire=expected_gain > float(features["probe_cost"]))
        raise AssertionError(f"unsupported simple policy {self.name}")

    def metadata(self) -> Mapping[str, Any]:
        return {
            "policy": self.name,
            "uses_oracle": False,
            "cache_policy": "disabled",
            "random_activation_rate": self.random_activation_rate
            if self.name == "random_budget_matched"
            else None,
        }


@dataclass(frozen=True)
class BoundedRolloutVoiPolicyAdapter:
    budget: int
    name: Literal["generic_bounded_rollout_voi"] = "generic_bounded_rollout_voi"

    def decide(
        self,
        *,
        features: Mapping[str, Any],
        seed: int,
        rollout: Callable[..., dict[str, Any]],
        probe_already_acquired: bool,
    ) -> PolicyDecision:
        _ = features
        if probe_already_acquired:
            return PolicyDecision(acquire=False)
        acquire, rollouts, branches = _bounded_generic_voi(
            rollout=rollout, seed=int(seed), budget=self.budget
        )
        return PolicyDecision(
            acquire=acquire, rollout_count=rollouts, branch_count=branches
        )

    def metadata(self) -> Mapping[str, Any]:
        return {
            "policy": self.name,
            "uses_oracle": False,
            "cache_policy": "disabled",
            "rollout_budget": self.budget,
        }


@dataclass(frozen=True)
class OraclePolicyAdapter:
    """Evaluator-only upper bound; it evaluates both counterfactual actions."""

    family: GeneratedCausalQuotientFamily
    name: Literal["oracle"] = "oracle"

    def decide(
        self,
        *,
        features: Mapping[str, Any],
        seed: int,
        rollout: Callable[..., dict[str, Any]],
        probe_already_acquired: bool,
    ) -> PolicyDecision:
        _ = features, rollout
        if probe_already_acquired:
            return PolicyDecision(acquire=False, uses_oracle=True)
        return PolicyDecision(
            acquire=_paired_empirical_gain(self.family, seed=int(seed)) > 0.0,
            uses_oracle=True,
        )

    def metadata(self) -> Mapping[str, Any]:
        return {
            "policy": self.name,
            "uses_oracle": True,
            "cache_policy": "not_applicable",
            "selection_eligible": False,
        }


@dataclass(frozen=True)
class ConditionalEfficiencyPilotResult:
    rows: tuple[dict[str, Any], ...]
    aggregate: tuple[dict[str, Any], ...]
    models: dict[str, GateModel]
    artifacts: dict[str, str]


@dataclass(frozen=True)
class ConditionalEfficiencyConfirmationConfig:
    """Frozen design for topology-held-out return--compute confirmation."""

    training_topology_families: tuple[str, ...] = (
        "positive_boundary_advantage",
        "null_or_competitive_endpoint",
    )
    heldout_topology_families: tuple[str, ...] = (
        "qualifying_or_negative_boundary_scope",
        "observability_scope_qualifier",
    )
    regimes: tuple[Regime, ...] = ("positive", "null", "invalid")
    distractor_counts: tuple[int, ...] = (0, 6, 12)
    horizons: tuple[int, ...] = (3, 5)
    probe_costs: tuple[float, ...] = (0.1, 0.4)
    generic_rollout_budgets: tuple[int, ...] = (1, 4, 16)
    primary_generic_rollout_budget: int = 16
    topology_instances_per_family: int = 12
    noninferiority_margin: float = 0.05
    cluster_bootstrap_repetitions: int = 1_000
    policy_arms: tuple[PolicyName, ...] = POLICIES
    cache_policy: str = "disabled_for_all_arms"
    training_seeds: tuple[int, ...] = tuple(range(101, 117))
    evaluation_seeds: tuple[int, ...] = tuple(range(401, 417))
    output_dir: str = "artifacts/causal_quotient_confirmation_matrix"

    def __post_init__(self) -> None:
        if set(self.training_topology_families) & set(self.heldout_topology_families):
            raise ValueError("training and held-out topology families must be disjoint")
        if not self.heldout_topology_families or not self.generic_rollout_budgets:
            raise ValueError(
                "confirmation requires held-out families and rollout budgets"
            )
        if any(value < 1 for value in self.horizons) or any(
            value < 0 for value in self.distractor_counts
        ):
            raise ValueError("horizons and distractor counts must be valid")
        if self.primary_generic_rollout_budget not in self.generic_rollout_budgets:
            raise ValueError(
                "primary generic rollout budget must be one of generic_rollout_budgets"
            )
        if self.topology_instances_per_family < 2:
            raise ValueError(
                "confirmation requires at least two topology instances per family"
            )
        if self.noninferiority_margin < 0.0 or self.cluster_bootstrap_repetitions < 1:
            raise ValueError(
                "non-inferiority margin and bootstrap repetitions must be valid"
            )
        if set(self.policy_arms) - set(POLICIES):
            raise ValueError("confirmation requested an unknown policy arm")


@dataclass(frozen=True)
class ConditionalEfficiencyConfirmationResult:
    rows: tuple[dict[str, Any], ...]
    aggregate: tuple[dict[str, Any], ...]
    models: dict[str, GateModel]
    artifacts: dict[str, str]


def build_pilot_specs() -> tuple[CausalQuotientFamilySpec, ...]:
    """Small paired positive/null/invalid panel with increasing distractors."""
    panel: tuple[tuple[Regime, int], ...] = (
        ("positive", 0),
        ("positive", 6),
        ("positive", 12),
        ("null", 0),
        ("null", 6),
        ("null", 12),
        ("invalid", 6),
    )
    return tuple(
        CausalQuotientFamilySpec(
            regime=regime,
            horizon=3,
            reward_delay=2,
            continuation_dependency_depth=3,
            distractor_count=distractors,
            probe_cost=0.1 if regime != "invalid" else 0.6,
            probe_reliability=1.0,
            sensing_budget=1,
            alias_persistence=3,
            seed=700 + index,
        )
        for index, (regime, distractors) in enumerate(panel)
    )


def run_conditional_efficiency_pilot(
    specs: Sequence[CausalQuotientFamilySpec] | None = None,
    *,
    config: ConditionalEfficiencyPilotConfig | None = None,
) -> ConditionalEfficiencyPilotResult:
    """Fit on training seeds and evaluate paired held-out seeds.

    The generic arm uses bounded model rollouts and enumerates ambient
    observation branches up to the same fixed rollout budget.  Its declared
    rollout/branch counts are primary compute accounting; wall-clock timing is
    merely diagnostic.
    """
    resolved_config = config or ConditionalEfficiencyPilotConfig()
    families = tuple(
        generate_causal_quotient_family(spec=spec, index=index)
        for index, spec in enumerate(specs or build_pilot_specs())
    )
    _reject_oracle_inputs(families)
    quotient_model = _fit_gate(
        families, seeds=resolved_config.training_seeds, uses_quotient=True
    )
    ablation_model = _fit_gate(
        families, seeds=resolved_config.training_seeds, uses_quotient=False
    )
    random_rate = _learned_activation_rate(families, quotient_model)
    rows: list[dict[str, Any]] = []
    for family in families:
        for seed in resolved_config.evaluation_seeds:
            for adapter in _policy_adapters(
                family,
                quotient_model=quotient_model,
                ablation_model=ablation_model,
                random_activation_rate=random_rate,
                rollout_budget=resolved_config.rollout_budget,
                policy_arms=POLICIES,
            ).values():
                rows.append(_run_episode(family, adapter=adapter, seed=int(seed)))
    aggregate = tuple(_aggregate(rows))
    artifact_paths = _write_artifacts(
        rows,
        aggregate,
        config=resolved_config,
        models={
            "learned_causal_quotient": quotient_model,
            "capacity_matched_no_quotient": ablation_model,
        },
    )
    result = ConditionalEfficiencyPilotResult(
        rows=tuple(rows),
        aggregate=aggregate,
        models={
            "learned_causal_quotient": quotient_model,
            "capacity_matched_no_quotient": ablation_model,
        },
        artifacts=artifact_paths,
    )
    return result


def build_confirmation_specs(
    config: ConditionalEfficiencyConfirmationConfig,
) -> tuple[CausalQuotientFamilySpec, ...]:
    """Materialize a factorial panel whose topology split is explicit."""
    specs: list[CausalQuotientFamilySpec] = []
    for topology_index, topology_family in enumerate(
        (*config.training_topology_families, *config.heldout_topology_families)
    ):
        for instance_index in range(config.topology_instances_per_family):
            instance_id = f"{topology_family}:instance-{instance_index:03d}"
            base_seed = 10_000 + topology_index * 10_000 + instance_index
            for regime in config.regimes:
                for distractors in config.distractor_counts:
                    for horizon in config.horizons:
                        for cost in config.probe_costs:
                            specs.append(
                                CausalQuotientFamilySpec(
                                    regime=regime,
                                    horizon=horizon,
                                    reward_delay=horizon - 1,
                                    continuation_dependency_depth=horizon,
                                    distractor_count=distractors,
                                    probe_cost=cost,
                                    probe_reliability=1.0,
                                    sensing_budget=1,
                                    alias_persistence=horizon,
                                    topology_family=topology_family,
                                    topology_instance_id=instance_id,
                                    seed=base_seed,
                                )
                            )
    return tuple(specs)


def run_conditional_efficiency_confirmation(
    *,
    config: ConditionalEfficiencyConfirmationConfig | None = None,
) -> ConditionalEfficiencyConfirmationResult:
    """Run the locked topology-held-out matrix without logging a claim outcome.

    All learned gates train only on the declared training topology families.
    Evaluation rows are exclusively held-out topologies.  Generic VoI is run
    at every fixed rollout budget, yielding a return--compute curve rather
    than a single potentially cherry-picked planning budget.
    """
    resolved_config = config or ConditionalEfficiencyConfirmationConfig()
    families = tuple(
        generate_causal_quotient_family(spec=spec, index=index)
        for index, spec in enumerate(build_confirmation_specs(resolved_config))
    )
    _reject_oracle_inputs(families)
    train = tuple(
        family
        for family in families
        if family.deployment_manifest["protocol"]["topology_family"]
        in resolved_config.training_topology_families
    )
    heldout = tuple(
        family
        for family in families
        if family.deployment_manifest["protocol"]["topology_family"]
        in resolved_config.heldout_topology_families
    )
    if not train or not heldout:
        raise ValueError(
            "confirmation split did not materialize both topology partitions"
        )
    quotient_model = _fit_gate(
        train, seeds=resolved_config.training_seeds, uses_quotient=True
    )
    ablation_model = _fit_gate(
        train, seeds=resolved_config.training_seeds, uses_quotient=False
    )
    random_rate = _learned_activation_rate(train, quotient_model)
    rows: list[dict[str, Any]] = []
    for family in heldout:
        protocol = family.deployment_manifest["protocol"]
        for seed in resolved_config.evaluation_seeds:
            base_adapters = _policy_adapters(
                family,
                quotient_model=quotient_model,
                ablation_model=ablation_model,
                random_activation_rate=random_rate,
                rollout_budget=resolved_config.primary_generic_rollout_budget,
                policy_arms=tuple(
                    arm
                    for arm in resolved_config.policy_arms
                    if arm != "generic_bounded_rollout_voi"
                ),
            )
            for adapter in base_adapters.values():
                row = _run_episode(family, adapter=adapter, seed=int(seed))
                rows.append(
                    _confirmation_row(row, protocol=protocol, generic_rollout_budget=0)
                )
            if "generic_bounded_rollout_voi" in resolved_config.policy_arms:
                for budget in resolved_config.generic_rollout_budgets:
                    rollout_adapter = BoundedRolloutVoiPolicyAdapter(budget=budget)
                    row = _run_episode(family, adapter=rollout_adapter, seed=int(seed))
                    rows.append(
                        _confirmation_row(
                            row, protocol=protocol, generic_rollout_budget=int(budget)
                        )
                    )
    aggregate = tuple(_aggregate_confirmation(rows))
    noninferiority = _confirmation_noninferiority(rows, config=resolved_config)
    artifacts = _write_confirmation_artifacts(
        rows,
        aggregate,
        config=resolved_config,
        models={
            "learned_causal_quotient": quotient_model,
            "capacity_matched_no_quotient": ablation_model,
        },
        noninferiority=noninferiority,
    )
    result = ConditionalEfficiencyConfirmationResult(
        tuple(rows),
        aggregate,
        {
            "learned_causal_quotient": quotient_model,
            "capacity_matched_no_quotient": ablation_model,
        },
        artifacts,
    )
    return result


def _fit_gate(
    families: Sequence[GeneratedCausalQuotientFamily],
    *,
    seeds: Iterable[int],
    uses_quotient: bool,
) -> GateModel:
    gains: dict[bool, list[float]] = defaultdict(list)
    gains_by_cost: dict[tuple[bool, float], list[float]] = defaultdict(list)
    for family in families:
        feature = _declared_query_disagreement(family) if uses_quotient else False
        cost = float(family.deployment_manifest["protocol"]["probe_cost"])
        for seed in seeds:
            gain = _paired_empirical_gain(family, seed=int(seed))
            gains[feature].append(gain)
            gains_by_cost[(feature, cost)].append(gain)
    costs = sorted(
        {
            float(family.deployment_manifest["protocol"]["probe_cost"])
            for family in families
        }
    )
    fallback = fmean(value for values in gains.values() for value in values)
    return GateModel(
        positive_gain=fmean(gains[True]) if gains[True] else fmean(gains[False]),
        nonpositive_gain=fmean(gains[False]),
        uses_quotient=uses_quotient,
        gain_by_feature_cost=tuple(
            (feature, cost, fmean(gains_by_cost.get((feature, cost), [fallback])))
            for feature in (False, True)
            for cost in costs
        ),
    )


def _paired_empirical_gain(
    family: GeneratedCausalQuotientFamily, *, seed: int
) -> float:
    return _fixed_acquisition_return(
        family, seed=seed, acquire=True
    ) - _fixed_acquisition_return(family, seed=seed, acquire=False)


def _learned_activation_rate(
    families: Sequence[GeneratedCausalQuotientFamily], model: GateModel
) -> float:
    if not families:
        return 0.0
    return fmean(
        float(
            model.acquire(
                query_disagreement=bool(
                    _controller_features(family)["query_disagreement"]
                ),
                probe_cost=float(_controller_features(family)["probe_cost"]),
            )
        )
        for family in families
    )


def _policy_adapters(
    family: GeneratedCausalQuotientFamily,
    *,
    quotient_model: GateModel,
    ablation_model: GateModel,
    random_activation_rate: float,
    rollout_budget: int,
    policy_arms: Sequence[PolicyName],
) -> dict[PolicyName, InformationPolicy]:
    """Materialize every arm through one auditable adapter boundary."""
    all_adapters: dict[PolicyName, InformationPolicy] = {
        "learned_causal_quotient": GatePolicyAdapter(
            "learned_causal_quotient", quotient_model
        ),
        "capacity_matched_no_quotient": GatePolicyAdapter(
            "capacity_matched_no_quotient", ablation_model
        ),
        "never_refine": SimplePolicyAdapter("never_refine"),
        "always_refine": SimplePolicyAdapter("always_refine"),
        "random_budget_matched": SimplePolicyAdapter(
            "random_budget_matched", random_activation_rate=random_activation_rate
        ),
        "uncertainty": SimplePolicyAdapter("uncertainty"),
        "information_gain": SimplePolicyAdapter("information_gain"),
        "generic_bounded_rollout_voi": BoundedRolloutVoiPolicyAdapter(
            budget=rollout_budget
        ),
        "oracle": OraclePolicyAdapter(family),
    }
    return {arm: all_adapters[arm] for arm in policy_arms}


def _confirmation_row(
    row: Mapping[str, Any], *, protocol: Mapping[str, Any], generic_rollout_budget: int
) -> dict[str, Any]:
    return {
        **row,
        "split": "heldout_topology",
        "topology_family": str(protocol["topology_family"]),
        "topology_instance_id": str(protocol["topology_instance_id"]),
        "horizon": int(protocol["horizon"]),
        "probe_cost": float(protocol["probe_cost"]),
        "generic_rollout_budget": int(generic_rollout_budget),
    }


def _confirmation_noninferiority(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: ConditionalEfficiencyConfirmationConfig,
) -> list[dict[str, Any]]:
    """Compare quotient return to paired non-oracle arms at topology-cluster level."""
    index: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        key = (
            row["topology_instance_id"],
            row["family_id"],
            row["seed"],
            row["regime"],
            row["distractor_count"],
            row["horizon"],
            row["probe_cost"],
            row["generic_rollout_budget"],
            row["policy"],
        )
        index[key] = row
    results: list[dict[str, Any]] = []
    comparators = [
        arm
        for arm in config.policy_arms
        if arm not in {"learned_causal_quotient", "oracle"}
    ]
    for comparator in comparators:
        budget = (
            config.primary_generic_rollout_budget
            if comparator == "generic_bounded_rollout_voi"
            else 0
        )
        differences: list[float] = []
        clusters: list[str] = []
        for row in rows:
            if (
                row["policy"] != "learned_causal_quotient"
                or int(row["generic_rollout_budget"]) != 0
            ):
                continue
            key = (
                row["topology_instance_id"],
                row["family_id"],
                row["seed"],
                row["regime"],
                row["distractor_count"],
                row["horizon"],
                row["probe_cost"],
                budget,
                comparator,
            )
            other = index.get(key)
            if other is None:
                continue
            differences.append(
                float(row["cost_adjusted_return"])
                - float(other["cost_adjusted_return"])
            )
            clusters.append(str(row["topology_instance_id"]))
        if differences:
            results.append(
                {
                    "candidate": "learned_causal_quotient",
                    "comparator": comparator,
                    "generic_rollout_budget": budget,
                    **clustered_noninferiority(
                        differences,
                        clusters,
                        margin=config.noninferiority_margin,
                        bootstrap_repetitions=config.cluster_bootstrap_repetitions,
                        seed=17 + len(results),
                    ),
                }
            )
    return results


def _paper_projection_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: ConditionalEfficiencyConfirmationConfig,
) -> dict[str, Any]:
    """Produce the figure-compatible aggregate from the immutable row artifact."""
    index: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        index[
            (
                row["topology_instance_id"],
                row["family_id"],
                row["seed"],
                row["regime"],
                row["distractor_count"],
                row["horizon"],
                row["probe_cost"],
                row["generic_rollout_budget"],
                row["policy"],
            )
        ] = row
    comparisons: list[dict[str, Any]] = []
    for regime in ("positive", "null"):
        for budget in config.generic_rollout_budgets:
            pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
            for row in rows:
                if (
                    row["policy"] != "learned_causal_quotient"
                    or row["regime"] != regime
                ):
                    continue
                other = index.get(
                    (
                        row["topology_instance_id"],
                        row["family_id"],
                        row["seed"],
                        row["regime"],
                        row["distractor_count"],
                        row["horizon"],
                        row["probe_cost"],
                        budget,
                        "generic_bounded_rollout_voi",
                    )
                )
                if other is not None:
                    pairs.append((row, other))
            if not pairs:
                continue
            gaps = [
                float(left["cost_adjusted_return"])
                - float(right["cost_adjusted_return"])
                for left, right in pairs
            ]
            comparisons.append(
                {
                    "regime": regime,
                    "budget": int(budget),
                    "return_gap_quotient_minus_generic": fmean(gaps),
                    "quotient_better": sum(gap > 0.0 for gap in gaps),
                    "equal": sum(gap == 0.0 for gap in gaps),
                    "quotient_worse": sum(gap < 0.0 for gap in gaps),
                    "quotient_mean_rollouts": fmean(
                        float(left["rollout_count"]) for left, _ in pairs
                    ),
                    "generic_mean_rollouts": fmean(
                        float(right["rollout_count"]) for _, right in pairs
                    ),
                    "quotient_mean_branches": fmean(
                        float(left["ambient_branch_count"]) for left, _ in pairs
                    ),
                    "generic_mean_branches": fmean(
                        float(right["ambient_branch_count"]) for _, right in pairs
                    ),
                    "paired_evaluation_count": len(pairs),
                }
            )
    return {
        "schema_version": "causal_observation_reproduction.paper1.conditional_efficiency_summary.v1",
        "claim_id": CLAIM_ID,
        "source": {
            "evaluation_scope": "locked topology-instance held-out synthetic grid",
            "cluster_key": "topology_instance_id",
        },
        "scope": {
            "heldout_topology_families": len(config.heldout_topology_families),
            "topology_instances_per_family": config.topology_instances_per_family,
            "evaluation_seeds": len(config.evaluation_seeds),
            "generic_rollout_budgets": list(config.generic_rollout_budgets),
        },
        "comparisons": comparisons,
    }


def _fixed_acquisition_return(
    family: GeneratedCausalQuotientFamily, *, seed: int, acquire: bool
) -> float:
    env = SequentialQuotientSensingEnv(family, seed=seed)
    observation = env.reset(seed=seed)
    probe: int | None = None
    total = 0.0
    while True:
        observation = env.acquire(
            "probe_context"
            if acquire and observation["probe_available"] and probe is None
            else "coarse"
        )
        if observation["probe_value"] is not None:
            probe = int(observation["probe_value"])
        step = env.control(_continuation_action(family, probe))
        total += step.reward
        observation = step.observation
        if step.terminated or step.truncated:
            return total


def _run_episode(
    family: GeneratedCausalQuotientFamily, *, adapter: InformationPolicy, seed: int
) -> dict[str, Any]:
    env = SequentialQuotientSensingEnv(family, seed=seed)
    observation = env.reset(seed=seed)
    total, cost, probe_count, rollout_count, branch_count, decision_ns = (
        0.0,
        0.0,
        0,
        0,
        0,
        0,
    )
    probe: int | None = None
    # Construct the controller-visible features once. The projection does not
    # carry a regime label, latent context, scorer sidecar, or semantic world.
    controller_features = _controller_features(family)

    def public_rollout(*, rollout_seed: int, acquire: bool) -> dict[str, Any]:
        return _public_rollout(family, seed=rollout_seed, acquire=acquire)

    while True:
        start = time.perf_counter_ns()
        decision = adapter.decide(
            features={
                **controller_features,
                "probe_available": bool(observation["probe_available"]),
            },
            seed=seed + int(observation["step_index"]) * 10_000,
            rollout=public_rollout,
            probe_already_acquired=probe is not None,
        )
        acquire = decision.acquire
        rollout_count += decision.rollout_count
        branch_count += decision.branch_count
        decision_ns += time.perf_counter_ns() - start
        observation = env.acquire(
            "probe_context" if acquire and observation["probe_available"] else "coarse"
        )
        if observation["probe_value"] is not None:
            probe = int(observation["probe_value"])
        step = env.control(_continuation_action(family, probe))
        total += step.reward
        cost += step.acquisition_cost
        probe_count += int(step.probe_attempted)
        observation = step.observation
        if step.terminated or step.truncated:
            break
    return {
        "family_id": family.family_id,
        "regime": family.deployment_manifest["declared_regime"],
        "seed": seed,
        "policy": adapter.name,
        "distractor_count": int(
            family.deployment_manifest["protocol"]["distractor_count"]
        ),
        "cost_adjusted_return": round(total, 12),
        "acquisition_cost": round(cost, 12),
        "probe_count": probe_count,
        "rollout_count": rollout_count,
        "ambient_branch_count": branch_count,
        "decision_latency_ns": decision_ns,
        "cache_hits": 0,
        "cache_policy": str(adapter.metadata().get("cache_policy", "disabled")),
        "uses_oracle": bool(adapter.metadata().get("uses_oracle", False)),
        "policy_metadata": dict(adapter.metadata()),
    }


def _bounded_generic_voi(
    *,
    rollout: Any,
    seed: int,
    budget: int,
) -> tuple[bool, int, int]:
    """A model-agnostic, bounded rollout estimate of information value.

    This planner is given only a callback returning public rollout traces and
    returns.  It cannot inspect the generated family, its declared regime, a
    scorer sidecar, or latent context.  Its branch count is the number of
    distinct *observed* trace signatures actually evaluated, never a synthetic
    cardinality derived from a distractor-count knob.
    """
    paired = [
        (
            rollout(rollout_seed=seed + offset, acquire=True),
            rollout(rollout_seed=seed + offset, acquire=False),
        )
        for offset in range(int(budget))
    ]
    gains = [
        float(acquired["return"]) - float(coarse["return"])
        for acquired, coarse in paired
    ]
    branches = {
        tuple(
            tuple(float(value) for value in observation)
            for observation in trace["ambient_trace"]
        )
        for pair in paired
        for trace in pair
    }
    return fmean(gains) > 0.0, len(paired) * 2, len(branches)


def _controller_features(family: GeneratedCausalQuotientFamily) -> dict[str, Any]:
    """Project a family into the only inputs allowed to learned controllers."""
    protocol = dict(family.deployment_manifest.get("protocol") or {})
    projected = {
        "query_disagreement": _declared_query_disagreement(family),
        "probe_cost": float(protocol["probe_cost"]),
        "probe_reliability": float(protocol["probe_reliability"]),
        "observation_noise": float(protocol["observation_noise"]),
    }
    forbidden = {
        "declared_regime",
        "regime",
        "latent_context",
        "scorer_sidecar",
        "semantic_world",
    }
    if forbidden & set(projected):
        raise AssertionError("controller feature projection leaked a forbidden field")
    return projected


def _public_rollout(
    family: GeneratedCausalQuotientFamily, *, seed: int, acquire: bool
) -> dict[str, Any]:
    """Run an acquisition counterfactual and expose only public observations."""
    env = SequentialQuotientSensingEnv(family, seed=seed)
    observation = env.reset(seed=seed)
    ambient_trace = [
        tuple(float(value) for value in observation["ambient_observation"])
    ]
    probe: int | None = None
    total = 0.0
    while True:
        observation = env.acquire(
            "probe_context"
            if acquire and observation["probe_available"] and probe is None
            else "coarse"
        )
        ambient_trace.append(
            tuple(float(value) for value in observation["ambient_observation"])
        )
        if observation["probe_value"] is not None:
            probe = int(observation["probe_value"])
        step = env.control(_continuation_action(family, probe))
        total += step.reward
        observation = step.observation
        ambient_trace.append(
            tuple(float(value) for value in observation["ambient_observation"])
        )
        if step.terminated or step.truncated:
            return {"return": total, "ambient_trace": tuple(ambient_trace)}


def _declared_query_disagreement(family: GeneratedCausalQuotientFamily) -> bool:
    """Read the compact, deployment-visible causal-query declaration only."""
    query = dict(family.deployment_manifest.get("causal_query") or {})
    value = query.get("action_value_varies_across_declared_contexts")
    if not isinstance(value, bool):
        raise ValueError(
            "deployment manifest lacks a boolean causal-query disagreement declaration"
        )
    return value


def _continuation_action(
    family: GeneratedCausalQuotientFamily, probe: int | None
) -> int:
    """Use acquired information only when the declared continuation query needs it.

    This is a common continuation controller for all three acquisition arms.
    It is not the quotient gate: it is the policy whose expected continuation
    value a generic VoI method is also required to evaluate.
    """
    return int(probe or 0) if _declared_query_disagreement(family) else 0


def _reject_oracle_inputs(families: Sequence[GeneratedCausalQuotientFamily]) -> None:
    for family in families:
        if family.scorer_sidecar.get("selection_eligible", True):
            raise ValueError("scorer sidecar must be selection-ineligible")


def _aggregate(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (str(row["regime"]), int(row["distractor_count"]), str(row["policy"]))
        ].append(row)
    return [
        {
            "regime": regime,
            "distractor_count": distractors,
            "policy": policy,
            "episode_count": len(group),
            **{
                f"mean_{field}": round(fmean(float(row[field]) for row in group), 12)
                for field in (
                    "cost_adjusted_return",
                    "acquisition_cost",
                    "probe_count",
                    "rollout_count",
                    "ambient_branch_count",
                    "decision_latency_ns",
                    "cache_hits",
                )
            },
        }
        for (regime, distractors, policy), group in sorted(grouped.items())
    ]


def _aggregate_confirmation(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    dimensions = (
        "split",
        "topology_family",
        "topology_instance_id",
        "regime",
        "distractor_count",
        "horizon",
        "probe_cost",
        "generic_rollout_budget",
        "policy",
    )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in dimensions)].append(row)
    metrics = (
        "cost_adjusted_return",
        "acquisition_cost",
        "probe_count",
        "rollout_count",
        "ambient_branch_count",
        "decision_latency_ns",
        "cache_hits",
    )
    return [
        {
            **dict(zip(dimensions, key, strict=True)),
            "episode_count": len(group),
            **{
                f"mean_{metric}": round(fmean(float(row[metric]) for row in group), 12)
                for metric in metrics
            },
        }
        for key, group in sorted(
            grouped.items(), key=lambda item: tuple(str(value) for value in item[0])
        )
    ]


def _write_artifacts(
    rows: Sequence[dict[str, Any]],
    aggregate: Sequence[dict[str, Any]],
    *,
    config: ConditionalEfficiencyPilotConfig,
    models: dict[str, GateModel],
) -> dict[str, str]:
    del rows  # Retained for source-compatible internal call signatures.
    root = Path(config.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    metrics = root / "pilot_metrics.json"
    manifest = root / "pilot_manifest.json"
    evidence = root / "pilot_evidence.json"
    metrics.write_text(
        json.dumps(list(aggregate), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "config": asdict(config),
                "models": {key: asdict(value) for key, value in models.items()},
                "controller_feature_contract": {
                    "forbidden": [
                        "scorer_sidecar",
                        "latent_context",
                        "declared_regime",
                    ],
                    "causal_quotient_feature": "distinct declared contextual action sequences",
                    "ablation_feature": "constant only",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    evidence.write_text(
        json.dumps(
            {
                "claim_id": CLAIM_ID,
                "assessment": "inconclusive_pilot",
                "claim": "Under valid decision-relative quotient factorization and high decision-irrelevant observation burden, causal quotient sensing can match generic bounded rollout-VoI return with lower declared online rollout/branch cost.",
                "limitations": [
                    "synthetic generated families",
                    "pilot-size held-out seeds",
                    "generic comparator is a bounded rollout approximation, not exact Bayesian VoI",
                    "no confirmation claim",
                ],
                "metrics_artifact": str(metrics),
                "manifest_artifact": str(manifest),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "metrics": str(metrics),
        "manifest": str(manifest),
        "evidence": str(evidence),
    }


def _write_confirmation_artifacts(
    rows: Sequence[dict[str, Any]],
    aggregate: Sequence[dict[str, Any]],
    *,
    config: ConditionalEfficiencyConfirmationConfig,
    models: dict[str, GateModel],
    noninferiority: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    root = Path(config.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    metrics_path, rows_path, manifest_path, evidence_path = (
        root / "confirmation_metrics.json",
        root / "confirmation_rows.json",
        root / "locked_confirmation_manifest.json",
        root / "confirmation_evidence.json",
    )
    policy_path, features_path, split_path, inference_path, paper_projection_path = (
        root / "policy_metadata.json",
        root / "feature_schema.json",
        root / "split_manifest.json",
        root / "clustered_noninferiority.json",
        root / "conditional_efficiency_confirmation_summary.json",
    )
    rows_path.write_text(
        json.dumps(list(rows), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metrics_path.write_text(
        json.dumps(list(aggregate), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    policy_metadata = {str(row["policy"]): dict(row["policy_metadata"]) for row in rows}
    feature_schema = {
        "allowed_controller_features": [
            "query_disagreement",
            "probe_cost",
            "probe_reliability",
            "observation_noise",
            "probe_available",
        ],
        "forbidden_controller_features": [
            "declared_regime",
            "regime",
            "latent_context",
            "scorer_sidecar",
            "semantic_world",
        ],
        "oracle_rows_selection_eligible": False,
    }
    split_manifest = {
        "training_topology_families": list(config.training_topology_families),
        "heldout_topology_families": list(config.heldout_topology_families),
        "cluster_key": "topology_instance_id",
        "heldout_topology_instance_ids": sorted(
            {str(row["topology_instance_id"]) for row in rows}
        ),
    }
    policy_path.write_text(
        json.dumps(policy_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    features_path.write_text(
        json.dumps(feature_schema, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    split_path.write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    inference_path.write_text(
        json.dumps(list(noninferiority), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paper_projection_path.write_text(
        json.dumps(
            _paper_projection_summary(rows, config=config), indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    design = {
        "schema_version": CONFIRMATION_SCHEMA_VERSION,
        "claim_id": CLAIM_ID,
        "status": "locked_pending_confirmation",
        "config": asdict(config),
        "models": {key: value.metadata() for key, value in models.items()},
        "training_split": "topology family",
        "evaluation_split": "disjoint held-out topology family and topology instance",
        "cluster_key": "topology_instance_id",
        "primary_metrics": [
            "cost_adjusted_return",
            "rollout_count",
            "ambient_branch_count",
            "decision_latency_ns",
        ],
        "negative_controls": ["null", "invalid"],
        "noninferiority": {
            "margin": config.noninferiority_margin,
            "method": "paired topology-cluster bootstrap",
            "primary_generic_rollout_budget": config.primary_generic_rollout_budget,
        },
        "cache_policy": config.cache_policy,
        "interpretation": "Do not claim confirmation until a frozen matrix run is reviewed; generic VoI is a bounded planner approximation.",
    }
    design["design_hash"] = sha256(
        json.dumps(design, sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(design, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    evidence_path.write_text(
        json.dumps(
            {
                "schema_version": CONFIRMATION_SCHEMA_VERSION,
                "claim_id": CLAIM_ID,
                "assessment": "requires_frozen_review",
                "claim": "A valid quotient may match the declared bounded generic-VoI comparator's cost-adjusted return with fewer declared online evaluations in supported held-out regimes.",
                "limitations": [
                    "generic comparator is bounded rollout VoI, not exact VoI",
                    "no conclusion is valid until frozen-manifest review",
                ],
                "manifest_artifact": str(manifest_path),
                "metrics_artifact": str(metrics_path),
                "rows_artifact": str(rows_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "metrics": str(metrics_path),
        "rows": str(rows_path),
        "manifest": str(manifest_path),
        "evidence": str(evidence_path),
        "policy_metadata": str(policy_path),
        "feature_schema": str(features_path),
        "split_manifest": str(split_path),
        "clustered_noninferiority": str(inference_path),
        "paper_projection": str(paper_projection_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run causal-quotient sequential sensing experiments."
    )
    parser.add_argument(
        "--confirmation",
        action="store_true",
        help="Run the locked topology-held-out matrix.",
    )
    parser.add_argument(
        "--output-dir", default="", help="Override the default artifact directory."
    )
    parser.add_argument(
        "--topology-instances-per-family",
        type=int,
        default=0,
        help="Frozen number of independently generated topology clusters per family.",
    )
    parser.add_argument(
        "--noninferiority-margin",
        type=float,
        default=-1.0,
        help="Pre-specified cost-adjusted-return non-inferiority margin.",
    )
    args = parser.parse_args(argv)
    if args.confirmation:
        config = ConditionalEfficiencyConfirmationConfig(
            output_dir=str(
                args.output_dir or ConditionalEfficiencyConfirmationConfig().output_dir
            ),
            topology_instances_per_family=(
                int(args.topology_instances_per_family)
                if int(args.topology_instances_per_family) > 0
                else ConditionalEfficiencyConfirmationConfig().topology_instances_per_family
            ),
            noninferiority_margin=(
                float(args.noninferiority_margin)
                if float(args.noninferiority_margin) >= 0.0
                else ConditionalEfficiencyConfirmationConfig().noninferiority_margin
            ),
        )
        result = run_conditional_efficiency_confirmation(config=config)
        print(
            json.dumps(
                {
                    "row_count": len(result.rows),
                    "aggregate_count": len(result.aggregate),
                    "artifacts": result.artifacts,
                },
                sort_keys=True,
            )
        )
        return 0
    return parser.error("select --confirmation")


__all__ = [
    "CLAIM_ID",
    "POLICIES",
    "ConditionalEfficiencyConfirmationConfig",
    "ConditionalEfficiencyConfirmationResult",
    "ConditionalEfficiencyPilotConfig",
    "ConditionalEfficiencyPilotResult",
    "build_confirmation_specs",
    "build_pilot_specs",
    "run_conditional_efficiency_confirmation",
    "run_conditional_efficiency_pilot",
]


if __name__ == "__main__":
    raise SystemExit(main())
