"""Portable POSCM source-target transfer design and readiness analysis."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Literal, NoReturn

DecisionBand = Literal["likely_transferable", "sandbox_transfer", "treat_as_new_domain"]


def _raise_invalid(message: str) -> NoReturn:
    """Raise a single input-contract error from validation helpers."""

    raise ValueError(message)


@dataclass(frozen=True)
class PoscmTransferConfig:
    """Source-derived, local-only factors for the controlled transfer panel."""

    seed: int = 17
    archive_replicates: int = 1
    pair_limit: int = 4
    train_episodes: int = 5_000
    evaluation_episodes: int = 128

    def __post_init__(self) -> None:
        if (
            self.archive_replicates < 1
            or self.pair_limit < 1
            or self.train_episodes < 1
            or self.evaluation_episodes < 1
        ):
            _raise_invalid("replicate and pair limits must be positive")


@dataclass(frozen=True)
class PoscmWorldSpec:
    """Minimal semantic POSCM parameters required by the transfer design."""

    action_count: int
    gate_depth: int
    distractor_count: int
    seed: int
    required_action_sequence: tuple[int, ...]

    @property
    def solution_density(self) -> float:
        return 1.0 / float(self.action_count**self.gate_depth)

    @property
    def world_id(self) -> str:
        payload = asdict(self)
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return f"poscm-world-{digest[:16]}"


@dataclass(frozen=True)
class PoscmTransferPair:
    """A predeclared POSCM source-target perturbation pair."""

    pair_class: str
    source: PoscmWorldSpec
    target: PoscmWorldSpec
    perturbation_type: str
    preserved_invariants: tuple[str, ...]
    broken_invariants: tuple[str, ...]

    @property
    def pair_id(self) -> str:
        payload = {
            "pair_class": self.pair_class,
            "source": self.source.world_id,
            "target": self.target.world_id,
            "perturbation_type": self.perturbation_type,
            "preserved": self.preserved_invariants,
            "broken": self.broken_invariants,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return f"poscm-transfer-pair-{digest[:16]}"


@dataclass(frozen=True)
class PoscmReadiness:
    """The portable source-target readiness decomposition."""

    score: float
    decision_band: DecisionBand
    control_compatibility: float
    structural_compatibility: float
    observability_compatibility: float
    goal_compatibility: float
    negative_control_risk: float


@dataclass(frozen=True)
class PoscmPolicyEvaluation:
    """Observed local transfer outcomes from the portable tabular controller."""

    source_success_rate: float
    target_zero_shot_success_rate: float
    target_random_success_rate: float
    transfer_lift_vs_target_random: float
    source_mastered: bool
    policy_backend: Literal["tabular_q_learning"]


def generate_poscm_transfer_pairs(
    config: PoscmTransferConfig | None = None,
) -> tuple[PoscmTransferPair, ...]:
    """Generate the four controlled pair classes per declared replicate."""

    resolved = config or PoscmTransferConfig()
    pairs: list[PoscmTransferPair] = []
    for replicate in range(resolved.archive_replicates):
        seed = resolved.seed + 101 * replicate
        source = _world(action_count=2, gate_depth=3, distractor_count=1, seed=seed)
        same = _world(action_count=2, gate_depth=3, distractor_count=1, seed=seed + 6)
        deeper = _world(action_count=2, gate_depth=4, distractor_count=1, seed=seed)
        branching = _world(action_count=3, gate_depth=3, distractor_count=2, seed=seed)
        control_break = PoscmWorldSpec(
            action_count=source.action_count,
            gate_depth=source.gate_depth,
            distractor_count=source.distractor_count,
            seed=source.seed,
            required_action_sequence=tuple(
                (action + 1) % source.action_count
                for action in source.required_action_sequence
            ),
        )
        pairs.extend(
            (
                PoscmTransferPair(
                    "same_motif_parameter_perturbation",
                    source,
                    same,
                    "same_motif_new_seed",
                    ("action_count", "gate_depth", "trap_count", "motif_region"),
                    ("surface_graph_instance",),
                ),
                PoscmTransferPair(
                    "same_action_deeper_gate",
                    source,
                    deeper,
                    "gate_depth_plus_one",
                    ("action_count", "trap_count", "motif_region"),
                    ("gate_depth", "solution_density"),
                ),
                PoscmTransferPair(
                    "branching_change_same_motif_family",
                    source,
                    branching,
                    "action_branching_plus_one",
                    ("gate_depth", "motif_region"),
                    ("action_count", "trap_count", "solution_density"),
                ),
                PoscmTransferPair(
                    "graph_close_control_break",
                    source,
                    control_break,
                    "required_action_complement",
                    ("surface_graph_instance", "action_count", "gate_depth"),
                    ("required_action_sequence", "trap_prefixes"),
                ),
            )
        )
    return tuple(pairs[: resolved.pair_limit])


def poscm_transfer_readiness(pair: PoscmTransferPair) -> PoscmReadiness:
    """Apply the source readiness weighting to portable structural distances."""

    source, target = pair.source, pair.target
    control = float(source.required_action_sequence == target.required_action_sequence)
    structural_distance = (
        abs(source.action_count - target.action_count)
        / max(source.action_count, target.action_count)
        + abs(source.gate_depth - target.gate_depth)
        / max(source.gate_depth, target.gate_depth)
    ) / 2.0
    structural = 1.0 - structural_distance
    observability = 1.0 - abs(source.distractor_count - target.distractor_count) / max(
        1, source.distractor_count, target.distractor_count
    )
    goal = 1.0 - max(
        abs(source.solution_density - target.solution_density),
        abs(source.gate_depth - target.gate_depth)
        / max(source.gate_depth, target.gate_depth),
    )
    risk = (
        1.0
        if structural >= 0.90 and control <= 0.70
        else 0.5
        if structural >= 0.80 and control <= 0.80
        else 0.0
    )
    score = max(
        0.0,
        min(
            1.0,
            0.45 * control
            + 0.20 * structural
            + 0.20 * observability
            + 0.15 * goal
            - 0.30 * risk,
        ),
    )
    return PoscmReadiness(
        score=score,
        decision_band=(
            "likely_transferable"
            if score >= 0.75
            else "sandbox_transfer"
            if score >= 0.50
            else "treat_as_new_domain"
        ),
        control_compatibility=control,
        structural_compatibility=structural,
        observability_compatibility=observability,
        goal_compatibility=goal,
        negative_control_risk=risk,
    )


def evaluate_poscm_transfer_pair(
    pair: PoscmTransferPair, config: PoscmTransferConfig | None = None
) -> PoscmPolicyEvaluation:
    """Train once on the source world and measure zero-shot target control.

    The environment is a finite controlled action-sequence POSCM projection:
    the controller observes its action prefix, receives terminal success only
    for the declared sequence, and is evaluated against a matched random
    policy.  This is intentionally a local policy backend, not a substitute
    for a managed deep-RL service.
    """

    resolved = config or PoscmTransferConfig()
    values = _train_tabular_policy(pair.source, episodes=resolved.train_episodes)
    source = _evaluate_tabular_policy(
        pair.source,
        values,
        episodes=resolved.evaluation_episodes,
        seed=pair.source.seed + 10_001,
    )
    target = _evaluate_tabular_policy(
        pair.target,
        values,
        episodes=resolved.evaluation_episodes,
        seed=pair.target.seed + 20_001,
    )
    random_target = _random_success_rate(pair.target)
    return PoscmPolicyEvaluation(
        source_success_rate=source,
        target_zero_shot_success_rate=target,
        target_random_success_rate=random_target,
        transfer_lift_vs_target_random=target - random_target,
        source_mastered=source >= 0.8,
        policy_backend="tabular_q_learning",
    )


def write_poscm_transfer_design(
    output_dir: str | Path, config: PoscmTransferConfig | None = None
) -> tuple[Path, Path]:
    """Write local POSCM pair definitions and structural readiness records."""

    resolved = config or PoscmTransferConfig()
    pairs = generate_poscm_transfer_pairs(resolved)
    rows = [
        {
            "pair_id": pair.pair_id,
            "pair_class": pair.pair_class,
            "perturbation_type": pair.perturbation_type,
            "preserved_invariants": pair.preserved_invariants,
            "broken_invariants": pair.broken_invariants,
            "source": asdict(pair.source),
            "target": asdict(pair.target),
            **asdict(poscm_transfer_readiness(pair)),
            **asdict(evaluate_poscm_transfer_pair(pair, resolved)),
        }
        for pair in pairs
    ]
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rows_path = directory / "poscm_transfer_pairs.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest_path = directory / "poscm_transfer_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "causal-observation-reproduction.poscm-transfer.v1",
                "config": asdict(resolved),
                "pair_count": len(rows),
                "policy_backend": "tabular_q_learning",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return rows_path, manifest_path


def _world(
    *, action_count: int, gate_depth: int, distractor_count: int, seed: int
) -> PoscmWorldSpec:
    rng = random.Random(seed)
    return PoscmWorldSpec(
        action_count,
        gate_depth,
        distractor_count,
        seed,
        tuple(rng.randrange(action_count) for _ in range(gate_depth)),
    )


def _train_tabular_policy(
    world: PoscmWorldSpec, *, episodes: int
) -> dict[tuple[int, ...], tuple[float, ...]]:
    values: dict[tuple[int, ...], list[float]] = {}
    rng = random.Random(world.seed)
    for episode in range(episodes):
        actions: list[int] = []
        visited: list[tuple[tuple[int, ...], int]] = []
        exploration = max(0.05, 1.0 - episode / min(1_000, episodes))
        for _ in range(world.gate_depth):
            state = tuple(actions)
            action_values = values.setdefault(state, [0.0] * world.action_count)
            action = (
                rng.randrange(world.action_count)
                if rng.random() < exploration
                else _best_action(action_values)
            )
            visited.append((state, action))
            actions.append(action)
        reward = float(tuple(actions) == world.required_action_sequence)
        for step, (state, action) in enumerate(reversed(visited)):
            target = (
                reward
                if step == 0
                else max(values[tuple(actions[: world.gate_depth - step])])
            )
            values[state][action] += 0.35 * (target - values[state][action])
    return {state: tuple(action_values) for state, action_values in values.items()}


def _evaluate_tabular_policy(
    world: PoscmWorldSpec,
    values: dict[tuple[int, ...], tuple[float, ...]],
    *,
    episodes: int,
    seed: int,
) -> float:
    del seed
    outcomes: list[float] = []
    for _ in range(episodes):
        actions: list[int] = []
        for _ in range(world.gate_depth):
            actions.append(
                _best_action(values.get(tuple(actions), (0.0,) * world.action_count))
            )
        outcomes.append(float(tuple(actions) == world.required_action_sequence))
    return fmean(outcomes)


def _random_success_rate(world: PoscmWorldSpec) -> float:
    return 1.0 / float(world.action_count**world.gate_depth)


def _best_action(values: tuple[float, ...] | list[float]) -> int:
    return max(range(len(values)), key=lambda action: values[action])


__all__ = [
    "DecisionBand",
    "PoscmPolicyEvaluation",
    "PoscmReadiness",
    "PoscmTransferConfig",
    "PoscmTransferPair",
    "PoscmWorldSpec",
    "evaluate_poscm_transfer_pair",
    "generate_poscm_transfer_pairs",
    "poscm_transfer_readiness",
    "write_poscm_transfer_design",
]
