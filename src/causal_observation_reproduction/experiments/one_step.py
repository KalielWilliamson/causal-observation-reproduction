"""Source-derived one-step COO calibration dataset and exact evaluator.

The experiment evaluates whether observable graph and cost features can gate a
coarse-versus-refine decision.  It intentionally materializes raw examples and
does not depend on a tracker, database, notebook runtime, or model framework.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Literal

Panel = Literal["positive", "null", "pathology"]
DatasetPanel = Literal[
    "development",
    "primary",
    "confirmation",
    "frozen_test",
    "independent_confirmation",
]

POSITIVE: Panel = "positive"
NULL: Panel = "null"
PATHOLOGY: Panel = "pathology"


@dataclass(frozen=True)
class OneStepWorld:
    """The finite, binary theorem-world used by a single calibration example."""

    seed: int
    state_one_probability: float
    reward_zero: float
    reward_one: float
    optimal_action_zero: int
    optimal_action_one: int
    observation_reliability: float
    observation_cost: float


@dataclass(frozen=True)
class OneStepEvaluation:
    """Exact and simulated values for one coarse/refined observation pair."""

    coarse_value: float
    fine_gross_value: float
    fine_net_value: float
    exact_gross_gap: float
    exact_net_gap: float
    empirical_gross_gap: float
    continuation_action_gap: float
    positive_reachability: bool
    incompatible_optimal_actions: bool


@dataclass(frozen=True)
class ObservableGraph:
    """The graph projection that is available to a learned gate."""

    motif: str
    nodes: tuple[tuple[str, str, bool], ...]
    edges: tuple[tuple[str, str], ...]

    def summary(self) -> dict[str, float]:
        """Return the observable graph features used by the calibration models."""

        roles = tuple(role for _, role, _ in self.nodes)
        return {
            "observable_node_count": float(len(self.nodes)),
            "observable_edge_count": float(len(self.edges)),
            "observable_action_count": float(roles.count("action")),
            "observable_mediator_count": float(roles.count("mediator")),
            "observable_proxy_count": float(roles.count("proxy")),
            "observable_outcome_count": float(roles.count("outcome")),
        }

    def json(self) -> str:
        """Serialize the public graph deterministically."""

        return json.dumps(
            {
                "nodes": [
                    {"id": node_id, "role": role, "query_member": query_member}
                    for node_id, role, query_member in self.nodes
                ],
                "edges": [
                    {"source": source, "target": target, "type": "causal"}
                    for source, target in self.edges
                ],
            },
            sort_keys=True,
        )


@dataclass(frozen=True)
class OneStepExample:
    """One observable input paired with evaluator-only COO targets."""

    dataset_panel: DatasetPanel
    panel: Panel
    graph: ObservableGraph
    world: OneStepWorld
    evaluation: OneStepEvaluation

    @property
    def instance_id(self) -> str:
        """Return the deterministic group identifier used by split checks."""

        return f"{self.dataset_panel}:{self.panel}:seed-{self.world.seed}"

    def as_record(self) -> dict[str, object]:
        """Return one JSON-serializable artifact row with label separation intact."""

        evaluation = self.evaluation
        metric_is_net = self.panel == PATHOLOGY
        raw_metric = (
            evaluation.exact_net_gap if metric_is_net else evaluation.exact_gross_gap
        )
        feature_values: dict[str, object] = {
            "observation_reliability": self.world.observation_reliability,
            "observation_cost": self.world.observation_cost,
            "probe_cost": 0.0,
            "observation_regime_coarse": 1.0,
            "query_continuation_action": 1.0,
            **self.graph.summary(),
        }
        region = {POSITIVE: "positive", NULL: "null", PATHOLOGY: "failure"}[self.panel]
        return {
            "schema_version": "causal-observation-reproduction.one-step.v1",
            "instance_id": self.instance_id,
            "base_graph_id": self.instance_id,
            "source_graph_id": f"observable-{self.graph.motif}",
            "target_graph_id": f"observable-{self.graph.motif}",
            "perturbation_family_id": self.panel,
            "motif_family": self.graph.motif,
            "generator_seed": self.world.seed,
            "query_id": "continuation_action",
            "observation_regime_id": "coarse",
            "experiment_panel": self.dataset_panel,
            "eligibility_status": "eligible",
            "graph_json": self.graph.json(),
            **feature_values,
            "V_coarse": evaluation.coarse_value,
            "V_refined": evaluation.fine_gross_value,
            "gross_refinement_value": evaluation.exact_gross_gap,
            "observation_cost_included_in_raw_metric": metric_is_net,
            "probe_cost_included_in_raw_metric": False,
            "net_refinement_value": evaluation.exact_net_gap,
            "oracle_action": "refine" if evaluation.exact_net_gap > 0.0 else "coarse",
            "registered_region": region,
            "sequential_value_gap_delta": raw_metric,
            "sequential_metric_name": (
                "sequential_value_gap_delta_pathology"
                if metric_is_net
                else "sequential_value_gap_delta"
            ),
            "rollout_count": 512,
            "continuation_action_gap": evaluation.continuation_action_gap,
            "positive_reachability": evaluation.positive_reachability,
            "incompatible_optimal_actions": evaluation.incompatible_optimal_actions,
        }


@dataclass(frozen=True)
class OneStepConfig:
    """The checked-in source configuration for the one-step materialization."""

    development_seeds: tuple[int, ...] = tuple(range(1001, 1081))
    frozen_test_seeds: tuple[int, ...] = tuple(range(2001, 2021))
    independent_confirmation_seeds: tuple[int, ...] = tuple(range(3001, 3101))
    primary_positive_seeds: tuple[int, ...] = tuple(range(1, 9))
    primary_null_seeds: tuple[int, ...] = tuple(range(1, 7))
    primary_pathology_seeds: tuple[int, ...] = tuple(range(1, 7))
    confirmation_positive_seeds: tuple[int, ...] = tuple(range(101, 109))
    confirmation_null_seeds: tuple[int, ...] = tuple(range(201, 207))
    rollouts: int = 512


def materialize_one_step_dataset(
    config: OneStepConfig | None = None,
) -> tuple[OneStepExample, ...]:
    """Materialize the exact source-configured calibration rows locally."""

    resolved = config or OneStepConfig()
    rows: list[OneStepExample] = []
    panels: tuple[tuple[DatasetPanel, Panel, tuple[int, ...]], ...] = (
        ("development", POSITIVE, resolved.development_seeds),
        ("development", NULL, resolved.development_seeds),
        ("development", PATHOLOGY, resolved.development_seeds),
        ("primary", POSITIVE, resolved.primary_positive_seeds),
        ("primary", NULL, resolved.primary_null_seeds),
        ("primary", PATHOLOGY, resolved.primary_pathology_seeds),
        ("confirmation", POSITIVE, resolved.confirmation_positive_seeds),
        ("confirmation", NULL, resolved.confirmation_null_seeds),
        ("frozen_test", POSITIVE, resolved.frozen_test_seeds),
        ("frozen_test", NULL, resolved.frozen_test_seeds),
        ("frozen_test", PATHOLOGY, resolved.frozen_test_seeds),
        (
            "independent_confirmation",
            POSITIVE,
            resolved.independent_confirmation_seeds,
        ),
        (
            "independent_confirmation",
            NULL,
            resolved.independent_confirmation_seeds,
        ),
        (
            "independent_confirmation",
            PATHOLOGY,
            resolved.independent_confirmation_seeds,
        ),
    )
    for dataset_panel, panel, seeds in panels:
        graph = observable_graph(panel)
        rows.extend(
            OneStepExample(
                dataset_panel=dataset_panel,
                panel=panel,
                graph=graph,
                world=build_one_step_world(panel, seed),
                evaluation=evaluate_one_step_world(
                    build_one_step_world(panel, seed), rollouts=resolved.rollouts
                ),
            )
            for seed in seeds
        )
    return tuple(sorted(rows, key=lambda row: row.instance_id))


def build_one_step_world(panel: Panel, seed: int) -> OneStepWorld:
    """Build a deterministic binary world for a declared design panel."""

    rng = random.Random(seed)
    probability = 0.35 + 0.30 * rng.random()
    reward_zero = 0.75 + 0.50 * rng.random()
    reward_one = 0.75 + 0.50 * rng.random()
    if panel == NULL:
        return OneStepWorld(seed, probability, reward_zero, reward_one, 0, 0, 1.0, 0.0)
    if panel == PATHOLOGY:
        return OneStepWorld(
            seed,
            probability,
            reward_zero,
            reward_one,
            0,
            1,
            0.55 + 0.10 * rng.random(),
            0.12 + 0.08 * rng.random(),
        )
    return OneStepWorld(seed, probability, reward_zero, reward_one, 0, 1, 1.0, 0.0)


def evaluate_one_step_world(
    world: OneStepWorld, *, rollouts: int = 512
) -> OneStepEvaluation:
    """Evaluate the frozen signal-conditioned controller and paired check.

    This intentionally follows the paper evaluator's declared continuation
    controller after a noisy signal. It does not re-optimize the action map
    from the posterior induced by that signal: doing so changes pathology rows
    and no longer reproduces the frozen experimental estimand.
    """

    probabilities = (1.0 - world.state_one_probability, world.state_one_probability)
    rewards = (world.reward_zero, world.reward_one)
    optimal_actions = (world.optimal_action_zero, world.optimal_action_one)
    coarse_action_values = tuple(
        sum(
            probability * reward
            for probability, reward, optimal_action in zip(
                probabilities, rewards, optimal_actions, strict=True
            )
            if action == optimal_action
        )
        for action in (0, 1)
    )
    coarse_value = max(coarse_action_values)
    perfect_refined_value = sum(
        probability * reward
        for probability, reward in zip(probabilities, rewards, strict=True)
    )
    fine_gross_value = world.observation_reliability * perfect_refined_value
    fine_net_value = fine_gross_value - world.observation_cost
    empirical_gap = _empirical_gross_gap(world, rollouts=rollouts)
    return OneStepEvaluation(
        coarse_value=coarse_value,
        fine_gross_value=fine_gross_value,
        fine_net_value=fine_net_value,
        exact_gross_gap=fine_gross_value - coarse_value,
        exact_net_gap=fine_net_value - coarse_value,
        empirical_gross_gap=empirical_gap,
        continuation_action_gap=min(world.reward_zero, world.reward_one),
        positive_reachability=0.0 < world.state_one_probability < 1.0,
        incompatible_optimal_actions=world.optimal_action_zero
        != world.optimal_action_one,
    )


def observable_graph(panel: Panel) -> ObservableGraph:
    """Return the exact observable graph projection for a motif family."""

    if panel == POSITIVE:
        return ObservableGraph(
            motif="boundary_mediator",
            nodes=(
                ("context", "context", False),
                ("boundary_root", "mediator", False),
                ("endpoint_outcome", "outcome", False),
            ),
            edges=(("context", "boundary_root"), ("boundary_root", "endpoint_outcome")),
        )
    if panel == NULL:
        return ObservableGraph(
            motif="endpoint_direct",
            nodes=(
                ("endpoint_action", "action", True),
                ("endpoint_outcome", "outcome", False),
            ),
            edges=(("endpoint_action", "endpoint_outcome"),),
        )
    return ObservableGraph(
        motif="collider_confounded",
        nodes=(
            ("boundary_process", "mediator", False),
            ("endpoint_outcome", "outcome", False),
            ("observed_collider", "proxy", False),
        ),
        edges=(
            ("boundary_process", "observed_collider"),
            ("endpoint_outcome", "observed_collider"),
        ),
    )


def write_one_step_dataset(
    output_dir: str | Path, config: OneStepConfig | None = None
) -> tuple[Path, Path]:
    """Write raw JSONL rows and a compact, source-configuration manifest."""

    resolved = config or OneStepConfig()
    rows = materialize_one_step_dataset(resolved)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    data_path = directory / "one_step_examples.jsonl"
    data_path.write_text(
        "".join(json.dumps(row.as_record(), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    frozen_count = sum(row.dataset_panel == "frozen_test" for row in rows)
    confirmation_count = sum(
        row.dataset_panel == "independent_confirmation" for row in rows
    )
    learning_count = len(rows) - frozen_count - confirmation_count
    manifest_path = directory / "one_step_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "causal-observation-reproduction.one-step.v1",
                "config": asdict(resolved),
                "row_count": len(rows),
                "frozen_test_row_count": frozen_count,
                "independent_confirmation_row_count": confirmation_count,
                "learning_row_count": learning_count,
                "paper_design_status": "learning_count_reconciled",
                "historical_protocol_modified": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return data_path, manifest_path


def _empirical_gross_gap(world: OneStepWorld, *, rollouts: int) -> float:
    rng = random.Random(world.seed * 104729 + 17)
    state_probabilities = (
        1.0 - world.state_one_probability,
        world.state_one_probability,
    )
    rewards = (world.reward_zero, world.reward_one)
    optimal_actions = (world.optimal_action_zero, world.optimal_action_one)
    coarse_values = tuple(
        sum(
            probability * reward
            for probability, reward, optimal_action in zip(
                state_probabilities, rewards, optimal_actions, strict=True
            )
            if action == optimal_action
        )
        for action in (0, 1)
    )
    coarse_action = 0 if coarse_values[0] >= coarse_values[1] else 1
    coarse_returns: list[float] = []
    fine_returns: list[float] = []
    for _ in range(max(1, rollouts)):
        state = 1 if rng.random() < world.state_one_probability else 0
        reward = rewards[state]
        optimal_action = optimal_actions[state]
        coarse_returns.append(reward if coarse_action == optimal_action else 0.0)
        observed_state = (
            state if rng.random() < world.observation_reliability else 1 - state
        )
        fine_action = optimal_actions[observed_state]
        fine_returns.append(reward if fine_action == optimal_action else 0.0)
    return fmean(fine_returns) - fmean(coarse_returns)


__all__ = [
    "NULL",
    "PATHOLOGY",
    "POSITIVE",
    "ObservableGraph",
    "OneStepConfig",
    "OneStepEvaluation",
    "OneStepExample",
    "OneStepWorld",
    "build_one_step_world",
    "evaluate_one_step_world",
    "materialize_one_step_dataset",
    "observable_graph",
    "write_one_step_dataset",
]
