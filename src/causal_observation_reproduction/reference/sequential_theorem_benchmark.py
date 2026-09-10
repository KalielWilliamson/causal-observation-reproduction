from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

POSITIVE_PANEL = "decision-relevant-refinement"
NULL_PANEL = "no-value-refinement"
ALIGNMENT_PANEL = "formal-empirical-alignment"
PATHOLOGY_PANEL = "pathology-failure-region"
SUPPORTED_PANELS = (
    POSITIVE_PANEL,
    NULL_PANEL,
    ALIGNMENT_PANEL,
    PATHOLOGY_PANEL,
)

PANEL_METRICS = {
    POSITIVE_PANEL: "sequential_value_gap_delta",
    NULL_PANEL: "sequential_value_gap_delta_null",
    ALIGNMENT_PANEL: "theorem_conditioned_value_gap_explained_variance",
    PATHOLOGY_PANEL: "sequential_value_gap_delta_pathology",
}


@dataclass(frozen=True)
class SequentialWorld:
    seed: int
    state_one_probability: float
    reward_zero: float
    reward_one: float
    optimal_action_zero: int
    optimal_action_one: int
    observation_reliability: float
    observation_cost: float


@dataclass(frozen=True)
class SequentialWorldResult:
    seed: int
    coarse_value: float
    fine_gross_value: float
    fine_net_value: float
    exact_gross_gap: float
    exact_net_gap: float
    empirical_gross_gap: float
    continuation_action_gap: float
    positive_reachability: bool
    incompatible_optimal_actions: bool
    deterministic_refinement: bool


def evaluate_world(
    world: SequentialWorld, *, rollouts: int = 512
) -> SequentialWorldResult:
    state_probabilities = (
        1.0 - world.state_one_probability,
        world.state_one_probability,
    )
    rewards = (world.reward_zero, world.reward_one)
    optimal_actions = (world.optimal_action_zero, world.optimal_action_one)

    coarse_action_values = tuple(
        sum(
            probability * reward
            for probability, reward, optimal_action in zip(
                state_probabilities,
                rewards,
                optimal_actions,
                strict=True,
            )
            if action == optimal_action
        )
        for action in (0, 1)
    )
    coarse_value = max(coarse_action_values)
    perfect_fine_value = sum(
        probability * reward
        for probability, reward in zip(state_probabilities, rewards, strict=True)
    )
    wrong_observation_value = sum(
        probability * reward
        for probability, reward, optimal_action in zip(
            state_probabilities,
            rewards,
            optimal_actions,
            strict=True,
        )
        if (1 - optimal_action) == optimal_action
    )
    fine_gross_value = (
        world.observation_reliability * perfect_fine_value
        + (1.0 - world.observation_reliability) * wrong_observation_value
    )
    fine_net_value = fine_gross_value - world.observation_cost
    exact_gross_gap = fine_gross_value - coarse_value
    exact_net_gap = fine_net_value - coarse_value

    rng = random.Random(world.seed * 104729 + 17)
    coarse_action = 0 if coarse_action_values[0] >= coarse_action_values[1] else 1
    coarse_returns: list[float] = []
    fine_returns: list[float] = []
    for _ in range(max(1, int(rollouts))):
        state = 1 if rng.random() < world.state_one_probability else 0
        reward = rewards[state]
        optimal_action = optimal_actions[state]
        coarse_returns.append(reward if coarse_action == optimal_action else 0.0)
        observed_state = (
            state if rng.random() < world.observation_reliability else 1 - state
        )
        fine_action = optimal_actions[observed_state]
        fine_returns.append(reward if fine_action == optimal_action else 0.0)
    empirical_gross_gap = fmean(fine_returns) - fmean(coarse_returns)

    return SequentialWorldResult(
        seed=world.seed,
        coarse_value=coarse_value,
        fine_gross_value=fine_gross_value,
        fine_net_value=fine_net_value,
        exact_gross_gap=exact_gross_gap,
        exact_net_gap=exact_net_gap,
        empirical_gross_gap=empirical_gross_gap,
        continuation_action_gap=min(rewards),
        positive_reachability=all(
            probability > 0.0 for probability in state_probabilities
        ),
        incompatible_optimal_actions=world.optimal_action_zero
        != world.optimal_action_one,
        deterministic_refinement=math.isclose(world.observation_reliability, 1.0),
    )


def build_world(panel: str, seed: int) -> SequentialWorld:
    rng = random.Random(int(seed))
    probability = 0.35 + 0.30 * rng.random()
    reward_zero = 0.75 + 0.50 * rng.random()
    reward_one = 0.75 + 0.50 * rng.random()
    if panel == NULL_PANEL:
        return SequentialWorld(
            seed=seed,
            state_one_probability=probability,
            reward_zero=reward_zero,
            reward_one=reward_one,
            optimal_action_zero=0,
            optimal_action_one=0,
            observation_reliability=1.0,
            observation_cost=0.0,
        )
    if panel == PATHOLOGY_PANEL:
        return SequentialWorld(
            seed=seed,
            state_one_probability=probability,
            reward_zero=reward_zero,
            reward_one=reward_one,
            optimal_action_zero=0,
            optimal_action_one=1,
            observation_reliability=0.55 + 0.10 * rng.random(),
            observation_cost=0.12 + 0.08 * rng.random(),
        )
    return SequentialWorld(
        seed=seed,
        state_one_probability=probability,
        reward_zero=reward_zero,
        reward_one=reward_one,
        optimal_action_zero=0,
        optimal_action_one=1,
        observation_reliability=1.0,
        observation_cost=0.0,
    )


def run_panel(
    panel: str,
    *,
    seeds: tuple[int, ...],
    rollouts: int = 512,
) -> dict[str, Any]:
    normalized_panel = str(panel).strip()
    if normalized_panel not in SUPPORTED_PANELS:
        raise ValueError(f"unsupported theorem panel: {panel}")
    results = tuple(
        evaluate_world(
            build_world(
                POSITIVE_PANEL
                if normalized_panel == ALIGNMENT_PANEL
                else normalized_panel,
                seed,
            ),
            rollouts=rollouts,
        )
        for seed in seeds
    )
    metric_key = PANEL_METRICS[normalized_panel]
    if normalized_panel == ALIGNMENT_PANEL:
        metric_value = _explained_variance(
            tuple(result.exact_gross_gap for result in results),
            tuple(result.empirical_gross_gap for result in results),
        )
    elif normalized_panel == PATHOLOGY_PANEL:
        metric_value = fmean(result.exact_net_gap for result in results)
    elif normalized_panel == NULL_PANEL:
        metric_value = fmean(result.exact_gross_gap for result in results)
    else:
        metric_value = fmean(result.exact_gross_gap for result in results)
    return {
        "schema_version": "causal_observation_reproduction.sequential_theorem_benchmark.v0",
        "panel": normalized_panel,
        "metric_key": metric_key,
        "metric_value": metric_value,
        metric_key: metric_value,
        "unit_count": len(results),
        "rollouts_per_unit": int(rollouts),
        "immediate_reward_identical": True,
        "future_transition_reward_only": True,
        "worlds": [asdict(result) for result in results],
    }


def _explained_variance(
    expected: tuple[float, ...], observed: tuple[float, ...]
) -> float:
    if len(expected) != len(observed) or not expected:
        return 0.0
    observed_mean = fmean(observed)
    total = sum((value - observed_mean) ** 2 for value in observed)
    if total <= 1e-12:
        return (
            1.0
            if all(
                math.isclose(a, b, abs_tol=1e-9)
                for a, b in zip(expected, observed, strict=True)
            )
            else 0.0
        )
    residual = sum(
        (actual - predicted) ** 2
        for predicted, actual in zip(expected, observed, strict=True)
    )
    return max(0.0, min(1.0, 1.0 - residual / total))


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run finite-horizon theorem-aligned observability panels."
    )
    parser.add_argument("--panel", choices=SUPPORTED_PANELS, required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--rollouts", type=int, default=512)
    parser.add_argument("--out-json", default="")
    args = parser.parse_args(argv)
    payload = run_panel(
        str(args.panel),
        seeds=_parse_seeds(str(args.seeds)),
        rollouts=max(1, int(args.rollouts)),
    )
    if args.out_json:
        out_path = Path(str(args.out_json))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {str(payload["metric_key"]): float(payload["metric_value"])}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
