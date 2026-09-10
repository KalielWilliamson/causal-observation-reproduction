"""Optional local DQN evaluation for the portable POSCM transfer panel.

The environment is the same finite action-sequence projection used by the
tabular transfer evaluator.  It deliberately has no tracker, checkpoint
service, or lab-world dependency; results are local evidence only and are not
historical-outcome parity claims.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from statistics import fmean
from typing import TYPE_CHECKING, Any, NoReturn

from causal_observation_reproduction.experiments.poscm_transfer import (
    PoscmTransferConfig,
    PoscmWorldSpec,
    generate_poscm_transfer_pairs,
    poscm_transfer_readiness,
)

if TYPE_CHECKING:
    from causal_observation_reproduction.experiments.poscm_transfer import (
        PoscmTransferPair,
    )


def _raise_invalid(message: str) -> NoReturn:
    raise ValueError(message)


@dataclass(frozen=True)
class PoscmDqnConfig:
    """Fixed local DQN protocol for the finite POSCM action-sequence task."""

    train_timesteps: int = 25_000
    evaluation_episodes: int = 128
    seed: int = 17
    learning_starts: int = 1_000
    source_mastery_threshold: float = 0.8

    def __post_init__(self) -> None:
        if self.train_timesteps < 1 or self.evaluation_episodes < 1:
            _raise_invalid("training and evaluation counts must be positive")
        if self.learning_starts < 0:
            _raise_invalid("learning_starts must be nonnegative")
        if not 0.0 <= self.source_mastery_threshold <= 1.0:
            _raise_invalid("source_mastery_threshold must lie in [0, 1]")


@dataclass(frozen=True)
class PoscmDqnEvaluation:
    """Source and zero-shot target outcomes from the portable DQN backend."""

    source_success_rate: float
    target_zero_shot_success_rate: float
    target_random_success_rate: float
    transfer_lift_vs_target_random: float
    source_mastered: bool
    policy_backend: str = "stable_baselines3_dqn"


def evaluate_poscm_dqn_pair(
    pair: PoscmTransferPair, config: PoscmDqnConfig | None = None
) -> PoscmDqnEvaluation:
    """Train a source DQN and measure zero-shot target success locally."""

    resolved = config or PoscmDqnConfig()
    gym, dqn = _load_optional_backend()
    observation_action_count = max(pair.source.action_count, pair.target.action_count)
    observation_gate_depth = max(pair.source.gate_depth, pair.target.gate_depth)
    source = _make_environment(
        gym,
        pair.source,
        seed=resolved.seed + pair.source.seed,
        observation_action_count=observation_action_count,
        observation_gate_depth=observation_gate_depth,
    )
    target = _make_environment(
        gym,
        pair.target,
        seed=resolved.seed + pair.target.seed,
        observation_action_count=observation_action_count,
        observation_gate_depth=observation_gate_depth,
    )
    try:
        model = dqn(
            "MlpPolicy",
            source,
            seed=resolved.seed + pair.source.seed,
            learning_starts=min(resolved.learning_starts, resolved.train_timesteps),
            verbose=0,
        )
        model.learn(total_timesteps=resolved.train_timesteps)
        source_success = _model_success_rate(
            source, model, episodes=resolved.evaluation_episodes, seed=resolved.seed
        )
        target_success = _model_success_rate(
            target,
            model,
            episodes=resolved.evaluation_episodes,
            seed=resolved.seed + 10_000,
        )
        target_random = _random_success_rate(
            target, episodes=resolved.evaluation_episodes, seed=resolved.seed + 20_000
        )
    finally:
        source.close()  # pylint: disable=no-member
        target.close()  # pylint: disable=no-member
    return PoscmDqnEvaluation(
        source_success_rate=source_success,
        target_zero_shot_success_rate=target_success,
        target_random_success_rate=target_random,
        transfer_lift_vs_target_random=target_success - target_random,
        source_mastered=source_success >= resolved.source_mastery_threshold,
    )


def write_poscm_dqn_evaluation(
    output_dir: str | Path,
    config: PoscmDqnConfig | None = None,
    *,
    transfer_config: PoscmTransferConfig | None = None,
) -> tuple[Path, Path]:
    """Evaluate every fixed POSCM transfer cell and write raw local evidence."""

    resolved = config or PoscmDqnConfig()
    pairs = generate_poscm_transfer_pairs(transfer_config)
    rows = []
    for pair in pairs:
        outcome = evaluate_poscm_dqn_pair(pair, resolved)
        rows.append(
            {
                "pair_id": pair.pair_id,
                "pair_class": pair.pair_class,
                "perturbation_type": pair.perturbation_type,
                "preserved_invariants": pair.preserved_invariants,
                "broken_invariants": pair.broken_invariants,
                "source": asdict(pair.source),
                "target": asdict(pair.target),
                **asdict(poscm_transfer_readiness(pair)),
                **asdict(outcome),
                "evidence_status": "local_replication_not_source_deep_rl_outcome_parity",
            }
        )
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rows_path = directory / "poscm_dqn_outcomes.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest_path = directory / "poscm_dqn_manifest.json"
    source_mastered_pair_count = sum(int(bool(row["source_mastered"])) for row in rows)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "causal-observation-reproduction.poscm-dqn.v1",
                "policy_config": asdict(resolved),
                "transfer_config": asdict(transfer_config or PoscmTransferConfig()),
                "pair_count": len(rows),
                "policy_backend": "stable_baselines3_dqn",
                "source_mastery": {
                    "threshold": resolved.source_mastery_threshold,
                    "source_mastered_pair_count": source_mastered_pair_count,
                    "all_source_pairs_mastered": source_mastered_pair_count
                    == len(rows),
                },
                "evidence_status": "local_replication_not_source_deep_rl_outcome_parity",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return rows_path, manifest_path


def _load_optional_backend() -> tuple[Any, Any]:
    try:
        gym = import_module("gymnasium")
        dqn = import_module("stable_baselines3").DQN
    except ImportError as error:
        message = "POSCM DQN evaluation requires `uv sync --extra rl`."
        raise RuntimeError(message) from error
    return gym, dqn


def _make_environment(
    gym: Any,
    world: PoscmWorldSpec,
    *,
    seed: int,
    observation_action_count: int,
    observation_gate_depth: int,
) -> Any:
    """Build a deterministic Gymnasium environment without a global registry."""

    action_count = world.action_count
    gate_depth = world.gate_depth
    if observation_action_count < action_count or observation_gate_depth < gate_depth:
        _raise_invalid("pair-level observation space must contain the local world")

    def init(self: Any) -> None:
        self.action_space = gym.spaces.Discrete(action_count)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(observation_gate_depth * observation_action_count + 1,),
            dtype="float32",
        )
        self.actions = []

    def reset(
        self: Any, *, seed: int | None = None, options: Any = None
    ) -> tuple[Any, dict[str, object]]:
        del seed, options
        self.actions = []
        return observation(self), {}

    def step(
        self: Any, action: int
    ) -> tuple[Any, float, bool, bool, dict[str, object]]:
        if not self.action_space.contains(action):
            message = f"action outside discrete action space: {action}"
            raise ValueError(message)
        self.actions.append(int(action))
        terminated = len(self.actions) == gate_depth
        reward = float(
            terminated and tuple(self.actions) == world.required_action_sequence
        )
        return observation(self), reward, terminated, False, {}

    def observation(self: Any) -> Any:
        vector = gym.spaces.utils.flatten_space(self.observation_space).sample() * 0.0
        for index, action in enumerate(self.actions):
            vector[index * observation_action_count + action] = 1.0
        vector[-1] = len(self.actions) / observation_gate_depth
        return vector

    environment_type = type(
        "PoscmActionSequenceEnv",
        (gym.Env,),
        {
            "metadata": {"render_modes": []},
            "__init__": init,
            "reset": reset,
            "step": step,
        },
    )
    environment = environment_type()
    environment.reset(seed=seed)
    return environment


def _model_success_rate(
    environment: Any, model: Any, *, episodes: int, seed: int
) -> float:
    outcomes: list[float] = []
    for episode in range(episodes):
        observation, _ = environment.reset(seed=seed + episode)
        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = environment.step(action)
            if terminated or truncated:
                outcomes.append(float(reward > 0.0))
                break
    return fmean(outcomes)


def _random_success_rate(environment: Any, *, episodes: int, seed: int) -> float:
    rng = random.Random(seed)
    outcomes: list[float] = []
    for episode in range(episodes):
        observation, _ = environment.reset(seed=seed + episode)
        del observation
        while True:
            action = rng.randrange(int(environment.action_space.n))
            _, reward, terminated, truncated, _ = environment.step(action)
            if terminated or truncated:
                outcomes.append(float(reward > 0.0))
                break
    return fmean(outcomes)


__all__ = [
    "PoscmDqnConfig",
    "PoscmDqnEvaluation",
    "evaluate_poscm_dqn_pair",
    "write_poscm_dqn_evaluation",
]
