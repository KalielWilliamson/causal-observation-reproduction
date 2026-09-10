"""Optional local DQN evaluator for a declared MiniGrid source-target pair."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from importlib import import_module
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Literal, NoReturn

from causal_observation_reproduction.experiments.minigrid import (
    MiniGridPairMatrixConfig,
    generate_minigrid_pair_matrix,
    structural_readiness,
)

if TYPE_CHECKING:
    from causal_observation_reproduction.experiments.minigrid import MiniGridPairSpec


def _raise_invalid(message: str) -> NoReturn:
    """Raise a single input-contract error from validation helpers."""

    raise ValueError(message)


@dataclass(frozen=True)
class MiniGridPolicyConfig:
    """Portable source DQN protocol with a fixed navigation interface."""

    train_timesteps: int = 50_000
    evaluation_episodes: int = 12
    seed: int = 17
    learning_starts: int = 500
    checkpoint_interval: int = 10_000
    source_mastery_threshold: float = 0.8
    observation_mode: Literal["symbolic_navigation"] = "symbolic_navigation"
    action_mode: Literal["navigation"] = "navigation"

    def __post_init__(self) -> None:
        if self.train_timesteps < 1 or self.evaluation_episodes < 1:
            _raise_invalid("training and evaluation counts must be positive")
        if self.learning_starts < 1 or self.checkpoint_interval < 1:
            _raise_invalid("learning start and checkpoint counts must be positive")
        if not 0.0 <= self.source_mastery_threshold <= 1.0:
            _raise_invalid("source mastery threshold must lie in [0, 1]")


@dataclass(frozen=True)
class MiniGridPolicyEvaluation:
    """Observed source and zero-shot target success for one frozen pair."""

    source_success_rate: float
    target_zero_shot_success_rate: float
    target_random_success_rate: float
    transfer_lift_vs_target_random: float
    source_mastered: bool
    policy_backend: str = "stable_baselines3_dqn"


@dataclass(frozen=True)
class MiniGridEvaluationConfig:
    """Full local pair-matrix execution protocol for the optional DQN backend."""

    pair_matrix: MiniGridPairMatrixConfig = field(
        default_factory=MiniGridPairMatrixConfig
    )
    policy: MiniGridPolicyConfig = field(default_factory=MiniGridPolicyConfig)

    def __post_init__(self) -> None:
        if self.pair_matrix.max_pairs < 1:
            _raise_invalid("the MiniGrid evaluation requires at least one pair")


def evaluate_minigrid_pair(
    pair: MiniGridPairSpec, config: MiniGridPolicyConfig | None = None
) -> MiniGridPolicyEvaluation:
    """Train DQN on source and evaluate it zero-shot on the matched target.

    Install the optional extra first: ``uv sync --extra minigrid``.  The
    function has no side effects beyond its in-memory local environments.
    """

    resolved = config or MiniGridPolicyConfig()
    gym, dqn, numpy = _load_optional_backend()
    source = _make_environment(
        gym,
        numpy,
        pair.source_env_id,
        seed=resolved.seed + pair.source_seed,
    )
    target = _make_environment(
        gym,
        numpy,
        pair.target_env_id,
        seed=resolved.seed + pair.target_seed,
    )
    try:
        model = dqn(
            "MlpPolicy",
            source,
            seed=resolved.seed + pair.source_seed,
            verbose=0,
            **_dqn_model_kwargs(resolved),
        )
        selected_model = _train_and_select_source_checkpoint(
            dqn, model, source, config=resolved
        )
        source_success = _model_success_rate(
            source,
            selected_model,
            episodes=resolved.evaluation_episodes,
            seed=resolved.seed,
        )
        target_success = _model_success_rate(
            target,
            selected_model,
            episodes=resolved.evaluation_episodes,
            seed=resolved.seed + 10_000,
        )
        target_random = _random_success_rate(
            target,
            episodes=resolved.evaluation_episodes,
            seed=resolved.seed + 20_000,
        )
    finally:
        source.close()  # pylint: disable=no-member
        target.close()  # pylint: disable=no-member
    return MiniGridPolicyEvaluation(
        source_success_rate=source_success,
        target_zero_shot_success_rate=target_success,
        target_random_success_rate=target_random,
        transfer_lift_vs_target_random=target_success - target_random,
        source_mastered=source_success >= resolved.source_mastery_threshold,
    )


def write_minigrid_policy_evaluation(
    output_dir: str | Path, config: MiniGridEvaluationConfig | None = None
) -> tuple[Path, Path]:
    """Evaluate each declared pair and write a local raw outcome table.

    The runner deliberately does not select pairs based on observed outcomes.
    It requires the ``minigrid`` extra and can take significant CPU time for
    the default 72-pair matrix.
    """

    resolved = config or MiniGridEvaluationConfig()
    pairs = generate_minigrid_pair_matrix(resolved.pair_matrix)
    rows = []
    for pair in pairs:
        outcome = evaluate_minigrid_pair(pair, resolved.policy)
        rows.append(
            {
                "pair_id": pair.pair_id,
                **asdict(pair),
                **asdict(
                    structural_readiness(
                        pair, threshold=resolved.pair_matrix.transfer_threshold
                    )
                ),
                **asdict(outcome),
                "evidence_status": "local_replication_pending_source_outcome_parity_review",
            }
        )
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rows_path = directory / "minigrid_policy_outcomes.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest_path = directory / "minigrid_policy_manifest.json"
    source_mastered_pair_count = sum(int(bool(row["source_mastered"])) for row in rows)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "causal-observation-reproduction.minigrid-policy.v1",
                "config": asdict(resolved),
                "pair_count": len(rows),
                "policy_backend": "stable_baselines3_dqn",
                "source_mastery": {
                    "threshold": resolved.policy.source_mastery_threshold,
                    "source_mastered_pair_count": source_mastered_pair_count,
                    "all_source_pairs_mastered": source_mastered_pair_count
                    == len(rows),
                },
                "evidence_status": "local_replication_pending_source_outcome_parity_review",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return rows_path, manifest_path


def _load_optional_backend() -> tuple[Any, Any, Any]:
    try:
        gym = import_module("gymnasium")
        import_module("minigrid")
        dqn = import_module("stable_baselines3").DQN
        numpy = import_module("numpy")
    except ImportError as error:
        message = "MiniGrid policy evaluation requires `uv sync --extra minigrid`."
        raise RuntimeError(message) from error
    return gym, dqn, numpy


def _make_environment(
    gym: Any,
    numpy: Any,
    env_id: str,
    *,
    seed: int,
) -> Any:
    """Build the fixed symbolic navigation interface for one MiniGrid task."""

    environment = gym.make(env_id)
    environment.reset(seed=seed)

    def action_init(self: Any, inner_environment: Any) -> None:
        gym.ActionWrapper.__init__(  # pylint: disable=unnecessary-dunder-call
            self, inner_environment
        )
        self.action_space = gym.spaces.Discrete(3)

    def action(self: Any, selected_action: int) -> int:
        del self
        return int(selected_action)

    action_wrapper_type = type(
        "NavigationActionWrapper",
        (gym.ActionWrapper,),
        {"__init__": action_init, "action": action},
    )
    navigation = action_wrapper_type(environment)

    def observation_init(self: Any, inner_environment: Any) -> None:
        gym.ObservationWrapper.__init__(  # pylint: disable=unnecessary-dunder-call
            self, inner_environment
        )
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(14,),
            dtype=numpy.float32,
        )

    def observation(self: Any, observation: Any) -> Any:
        del observation
        unwrapped = self.unwrapped
        width = max(1, int(getattr(unwrapped, "width", 1)) - 1)
        height = max(1, int(getattr(unwrapped, "height", 1)) - 1)
        x, y = getattr(unwrapped, "agent_pos", (0, 0))
        direction = int(getattr(unwrapped, "agent_dir", 0))
        goal_x, goal_y = _first_goal_position(unwrapped)
        delta_x = float(goal_x - int(x)) / float(width)
        delta_y = float(goal_y - int(y)) / float(height)
        values = (
            float(x) / float(width),
            float(y) / float(height),
            *(1.0 if direction == index else 0.0 for index in range(4)),
            float(goal_x) / float(width),
            float(goal_y) / float(height),
            delta_x,
            delta_y,
            abs(delta_x) + abs(delta_y),
            1.0 if _front_cell_blocked(unwrapped) else 0.0,
            1.0 if int(x) == goal_x and int(y) == goal_y else 0.0,
            float(getattr(unwrapped, "step_count", 0))
            / max(1.0, float(getattr(unwrapped, "max_steps", 1))),
        )
        return numpy.asarray(values, dtype=numpy.float32)

    observation_wrapper_type = type(
        "SymbolicNavigationObservationWrapper",
        (gym.ObservationWrapper,),
        {"__init__": observation_init, "observation": observation},
    )
    symbolic = observation_wrapper_type(navigation)
    symbolic.reset(seed=seed)  # pylint: disable=no-member
    return symbolic


def _dqn_model_kwargs(config: MiniGridPolicyConfig) -> dict[str, object]:
    return {
        "learning_starts": min(config.learning_starts, config.train_timesteps),
        "buffer_size": max(1_000, min(50_000, config.train_timesteps * 2)),
        "train_freq": 1,
        "gradient_steps": 1,
        "exploration_fraction": 0.35,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.02,
        "gamma": 0.98,
        "learning_rate": 1e-3,
        "batch_size": 32,
        "target_update_interval": 10_000,
    }


def _train_and_select_source_checkpoint(
    dqn: Any, model: Any, source: Any, *, config: MiniGridPolicyConfig
) -> Any:
    """Select the best source-success checkpoint without persisting a service."""

    remaining = config.train_timesteps
    best_success = -1.0
    selected_model = model
    with TemporaryDirectory(prefix="coo_minigrid_dqn_") as directory:
        checkpoint_path = Path(directory) / "best_source_policy.zip"
        while remaining > 0:
            interval = min(config.checkpoint_interval, remaining)
            model.learn(total_timesteps=interval, reset_num_timesteps=False)
            remaining -= interval
            checkpoint_success = _model_success_rate(
                source,
                model,
                episodes=min(4, config.evaluation_episodes),
                seed=config.seed,
            )
            if checkpoint_success >= best_success:
                best_success = checkpoint_success
                model.save(checkpoint_path)
        if checkpoint_path.is_file():
            selected_model = dqn.load(checkpoint_path, env=source)
    return selected_model


def _first_goal_position(environment: Any) -> tuple[int, int]:
    width = int(environment.width)
    height = int(environment.height)
    grid = environment.grid
    for x in range(width):
        for y in range(height):
            object_at_position = grid.get(x, y)
            if getattr(object_at_position, "type", "") == "goal":
                return x, y
    return width - 1, height - 1


def _front_cell_blocked(environment: Any) -> bool:
    x, y = getattr(environment, "agent_pos", (0, 0))
    direction = int(getattr(environment, "agent_dir", 0))
    direction_vectors = ((1, 0), (0, 1), (-1, 0), (0, -1))
    delta_x, delta_y = direction_vectors[direction % len(direction_vectors)]
    front_x, front_y = int(x) + delta_x, int(y) + delta_y
    if (
        front_x < 0
        or front_y < 0
        or front_x >= environment.width
        or front_y >= environment.height
    ):
        return True
    object_at_position = environment.grid.get(front_x, front_y)
    if object_at_position is None:
        return False
    object_type = getattr(object_at_position, "type", "")
    if object_type == "door":
        return not bool(getattr(object_at_position, "is_open", False))
    return object_type in {"wall", "lava"}


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
        _, _ = environment.reset(seed=seed + episode)
        while True:
            action = rng.randrange(int(environment.action_space.n))
            _, reward, terminated, truncated, _ = environment.step(action)
            if terminated or truncated:
                outcomes.append(float(reward > 0.0))
                break
    return fmean(outcomes)


__all__ = [
    "MiniGridEvaluationConfig",
    "MiniGridPolicyConfig",
    "MiniGridPolicyEvaluation",
    "evaluate_minigrid_pair",
    "write_minigrid_policy_evaluation",
]
