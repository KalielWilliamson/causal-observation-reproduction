"""Frozen PPO MiniGrid control panel used for outcome-parity review.

The implementation retains only the environment, policy, and evaluator parts
needed to reproduce the reviewed control panel.  It deliberately has no event
store, experiment tracker, or configuration framework.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, NoReturn

PolicyAlgorithm = Literal["ppo"]
ControlRole = Literal["positive_control", "near_neighbor", "negative_control"]


def _raise_invalid(message: str) -> NoReturn:
    raise ValueError(message)


@dataclass(frozen=True)
class PoscmMiniGridTask:
    """Small control-signature task embedded in the MiniGrid navigation API."""

    label: str
    size: int
    required_actions: tuple[int, ...]
    trap_prefixes: tuple[tuple[int, ...], ...]
    reward_delay: int

    def __post_init__(self) -> None:
        if self.size < 5:
            _raise_invalid("POSCM MiniGrid task size must be at least five")
        if not self.required_actions:
            _raise_invalid("POSCM MiniGrid task needs a required action sequence")
        if any(action not in {0, 1, 2} for action in self.required_actions):
            _raise_invalid("POSCM MiniGrid actions must use the navigation alphabet")
        if any(action not in {0, 1, 2} for row in self.trap_prefixes for action in row):
            _raise_invalid("POSCM MiniGrid traps must use the navigation alphabet")
        if self.reward_delay < 0:
            _raise_invalid("POSCM MiniGrid reward delay cannot be negative")


@dataclass(frozen=True)
class PaperMiniGridEnvironment:
    """Typed public description of a normal or control-signature environment."""

    env_id: str = ""
    poscm: PoscmMiniGridTask | None = None

    def __post_init__(self) -> None:
        if bool(self.env_id) == bool(self.poscm):
            _raise_invalid("an environment needs exactly one implementation")

    @property
    def identifier(self) -> str:
        if self.poscm is None:
            return self.env_id
        return f"poscm:{self.poscm.label}"


@dataclass(frozen=True)
class PaperMiniGridCell:
    """One predeclared PPO transfer control with its reference outcome."""

    cell_id: str
    source: PaperMiniGridEnvironment
    target: PaperMiniGridEnvironment
    source_seed: int
    target_seed: int
    perturbation_type: str
    control_role: ControlRole
    expected_transfer: bool


@dataclass(frozen=True)
class PaperMiniGridPpoConfig:
    """Frozen local PPO protocol for the reviewed MiniGrid control panel."""

    train_timesteps: int = 4_096
    evaluation_episodes: int = 6
    checkpoint_interval: int = 4_096
    source_mastery_threshold: float = 0.8
    transfer_effect_threshold: float = 0.25
    repeats: int = 2
    policy_algorithm: PolicyAlgorithm = "ppo"
    learning_rate: float = 3e-4
    n_steps: int = 256
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5

    def __post_init__(self) -> None:
        if self.train_timesteps < 1 or self.evaluation_episodes < 1:
            _raise_invalid("training and evaluation counts must be positive")
        if self.checkpoint_interval < 1 or self.repeats < 1:
            _raise_invalid("checkpoint and repeat counts must be positive")
        if self.n_steps < 2 or self.batch_size < 1 or self.n_epochs < 1:
            _raise_invalid("PPO rollout, batch, and epoch counts must be positive")
        if not 0.0 <= self.source_mastery_threshold <= 1.0:
            _raise_invalid("source mastery threshold must lie in [0, 1]")
        if not 0.0 <= self.transfer_effect_threshold <= 1.0:
            _raise_invalid("transfer effect threshold must lie in [0, 1]")


@dataclass(frozen=True)
class PaperMiniGridIndependentSeeds:
    """Independent optimizer, validation, and evaluation randomization."""

    optimizer_seed: int
    validation_seeds: tuple[int, ...]
    source_evaluation_seeds: tuple[int, ...]
    target_evaluation_seeds: tuple[int, ...]
    random_action_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.validation_seeds:
            _raise_invalid("independent validation seeds must be nonempty")
        if not self.source_evaluation_seeds or not self.target_evaluation_seeds:
            _raise_invalid("independent evaluation seeds must be nonempty")
        if len(self.target_evaluation_seeds) != len(self.random_action_seeds):
            _raise_invalid("target environment and random-action seeds must align")


@dataclass(frozen=True)
class PaperMiniGridOutcome:
    """Per-control outcome and the predeclared parity decision."""

    source_success_rate: float
    target_zero_shot_success_rate: float
    target_random_success_rate: float
    source_efficiency_score: float
    target_zero_shot_efficiency_score: float
    target_random_efficiency_score: float
    transfer_effect_score: float
    source_mastered: bool
    observed_transfer_success: bool
    agrees_with_reference: bool
    policy_backend: str = "stable_baselines3_ppo"


def paper_minigrid_cells() -> tuple[PaperMiniGridCell, ...]:
    """Return the frozen eight-control panel before outcome observation."""

    empty_five = PaperMiniGridEnvironment(env_id="MiniGrid-Empty-5x5-v0")
    empty_six = PaperMiniGridEnvironment(env_id="MiniGrid-Empty-6x6-v0")
    doorkey_five = PaperMiniGridEnvironment(env_id="MiniGrid-DoorKey-5x5-v0")
    poscm_source = PaperMiniGridEnvironment(
        poscm=PoscmMiniGridTask(
            label="signature-source",
            size=6,
            required_actions=(2, 2, 2),
            trap_prefixes=((0,), (1,)),
            reward_delay=1,
        )
    )
    poscm_preserved = PaperMiniGridEnvironment(
        poscm=PoscmMiniGridTask(
            label="signature-preserved-scale",
            size=8,
            required_actions=(2, 2, 2),
            trap_prefixes=((0,), (1,)),
            reward_delay=1,
        )
    )
    poscm_broken = PaperMiniGridEnvironment(
        poscm=PoscmMiniGridTask(
            label="signature-broken-control",
            size=6,
            required_actions=(1, 1, 1),
            trap_prefixes=((2,),),
            reward_delay=0,
        )
    )
    return (
        PaperMiniGridCell(
            "cross-family-seed-11",
            empty_five,
            doorkey_five,
            11,
            11,
            "cross_family_causal_negative_control",
            "negative_control",
            False,
        ),
        PaperMiniGridCell(
            "signature-control-break",
            poscm_source,
            poscm_broken,
            11,
            11,
            "poscm_seeded_control_break_negative_control",
            "negative_control",
            False,
        ),
        PaperMiniGridCell(
            "signature-preserved-scale",
            poscm_source,
            poscm_preserved,
            11,
            11,
            "poscm_seeded_control_preserved_scale_change",
            "near_neighbor",
            True,
        ),
        PaperMiniGridCell(
            "signature-identity",
            poscm_source,
            poscm_source,
            11,
            11,
            "poscm_seeded_same_control",
            "positive_control",
            True,
        ),
        PaperMiniGridCell(
            "identity-seed-11",
            empty_five,
            empty_five,
            11,
            11,
            "same_env_same_seed",
            "positive_control",
            True,
        ),
        PaperMiniGridCell(
            "seed-shift-11-to-17",
            empty_five,
            empty_five,
            11,
            17,
            "same_env_seed_shift",
            "positive_control",
            True,
        ),
        PaperMiniGridCell(
            "scale-5-to-6",
            empty_five,
            empty_six,
            11,
            11,
            "same_family_scale_change",
            "near_neighbor",
            True,
        ),
        PaperMiniGridCell(
            "cross-family-seed-17",
            empty_five,
            doorkey_five,
            17,
            17,
            "cross_family_causal_negative_control",
            "negative_control",
            False,
        ),
    )


def evaluate_paper_minigrid_cell(
    cell: PaperMiniGridCell,
    config: PaperMiniGridPpoConfig | None = None,
) -> PaperMiniGridOutcome:
    """Train one source PPO policy and evaluate its fixed target control."""

    resolved = config or PaperMiniGridPpoConfig()
    gym, numpy, ppo = _load_backend()
    source = _make_environment(gym, numpy, cell.source, seed=cell.source_seed)
    try:
        model = ppo(
            "MlpPolicy",
            source,
            seed=cell.source_seed,
            verbose=0,
            **_ppo_kwargs(resolved),
        )
        selected_model = _train_and_select_checkpoint(
            ppo, model, cell, config=resolved, gym=gym, numpy=numpy
        )
    finally:
        source.close()
    source_summary = _evaluate_policy(
        selected_model,
        cell.source,
        seed=cell.source_seed,
        config=resolved,
        gym=gym,
        numpy=numpy,
    )
    target_summary = _evaluate_policy(
        selected_model,
        cell.target,
        seed=cell.target_seed,
        config=resolved,
        gym=gym,
        numpy=numpy,
    )
    random_summary = _evaluate_random(
        cell.target, seed=cell.target_seed, config=resolved, gym=gym, numpy=numpy
    )
    transfer_effect = _normalized_transfer_effect(
        target_summary.efficiency_score, random_summary.efficiency_score
    )
    source_mastered = (
        source_summary.efficiency_score >= resolved.source_mastery_threshold
    )
    observed_transfer = (
        source_mastered and transfer_effect >= resolved.transfer_effect_threshold
    )
    return PaperMiniGridOutcome(
        source_success_rate=source_summary.success_rate,
        target_zero_shot_success_rate=target_summary.success_rate,
        target_random_success_rate=random_summary.success_rate,
        source_efficiency_score=source_summary.efficiency_score,
        target_zero_shot_efficiency_score=target_summary.efficiency_score,
        target_random_efficiency_score=random_summary.efficiency_score,
        transfer_effect_score=transfer_effect,
        source_mastered=source_mastered,
        observed_transfer_success=observed_transfer,
        agrees_with_reference=observed_transfer == cell.expected_transfer,
    )


def evaluate_paper_minigrid_cell_independent(
    cell: PaperMiniGridCell,
    seeds: PaperMiniGridIndependentSeeds,
    config: PaperMiniGridPpoConfig | None = None,
) -> PaperMiniGridOutcome:
    """Evaluate a cell with disjoint optimizer, validation, and test seeds."""

    resolved = config or PaperMiniGridPpoConfig()
    gym, numpy, ppo = _load_backend()
    source = _make_environment(gym, numpy, cell.source, seed=cell.source_seed)
    try:
        model = ppo(
            "MlpPolicy",
            source,
            seed=seeds.optimizer_seed,
            verbose=0,
            **_ppo_kwargs(resolved),
        )
        selected_model = _train_and_select_checkpoint(
            ppo,
            model,
            cell,
            config=resolved,
            gym=gym,
            numpy=numpy,
            validation_seeds=seeds.validation_seeds,
        )
    finally:
        source.close()
    source_summary = _evaluate_policy(
        selected_model,
        cell.source,
        seed=cell.source_seed,
        config=resolved,
        gym=gym,
        numpy=numpy,
        episode_seeds=seeds.source_evaluation_seeds,
    )
    target_summary = _evaluate_policy(
        selected_model,
        cell.target,
        seed=cell.target_seed,
        config=resolved,
        gym=gym,
        numpy=numpy,
        episode_seeds=seeds.target_evaluation_seeds,
    )
    random_summary = _evaluate_random(
        cell.target,
        seed=cell.target_seed,
        config=resolved,
        gym=gym,
        numpy=numpy,
        environment_seeds=seeds.target_evaluation_seeds,
        action_seeds=seeds.random_action_seeds,
    )
    transfer_effect = _normalized_transfer_effect(
        target_summary.efficiency_score, random_summary.efficiency_score
    )
    source_mastered = (
        source_summary.efficiency_score >= resolved.source_mastery_threshold
    )
    observed_transfer = (
        source_mastered and transfer_effect >= resolved.transfer_effect_threshold
    )
    return PaperMiniGridOutcome(
        source_success_rate=source_summary.success_rate,
        target_zero_shot_success_rate=target_summary.success_rate,
        target_random_success_rate=random_summary.success_rate,
        source_efficiency_score=source_summary.efficiency_score,
        target_zero_shot_efficiency_score=target_summary.efficiency_score,
        target_random_efficiency_score=random_summary.efficiency_score,
        transfer_effect_score=transfer_effect,
        source_mastered=source_mastered,
        observed_transfer_success=observed_transfer,
        agrees_with_reference=observed_transfer == cell.expected_transfer,
    )


def write_paper_minigrid_policy_evaluation(
    output_dir: str | Path,
    config: PaperMiniGridPpoConfig | None = None,
    cells: tuple[PaperMiniGridCell, ...] | None = None,
) -> tuple[Path, Path]:
    """Materialize every frozen PPO control and its parity decision."""

    resolved = config or PaperMiniGridPpoConfig()
    resolved_cells = cells or paper_minigrid_cells()
    if not resolved_cells:
        _raise_invalid("the paper MiniGrid panel must contain at least one cell")
    rows: list[dict[str, Any]] = []
    for cell in resolved_cells:
        for repeat_index in range(resolved.repeats):
            outcome = evaluate_paper_minigrid_cell(cell, resolved)
            rows.append(
                {
                    "cell_id": cell.cell_id,
                    "repeat_index": repeat_index,
                    "source_environment": cell.source.identifier,
                    "target_environment": cell.target.identifier,
                    "source_seed": cell.source_seed,
                    "target_seed": cell.target_seed,
                    "perturbation_type": cell.perturbation_type,
                    "control_role": cell.control_role,
                    "expected_transfer": cell.expected_transfer,
                    **asdict(outcome),
                    "policy_algorithm": resolved.policy_algorithm,
                    "evidence_status": "local_outcome_parity_review",
                }
            )
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rows_path = directory / "paper_minigrid_ppo_outcomes.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    agreement_count = sum(int(bool(row["agrees_with_reference"])) for row in rows)
    manifest_path = directory / "paper_minigrid_ppo_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "causal-observation-reproduction.paper-minigrid-ppo.v1",
                "config": asdict(resolved),
                "logical_cell_count": len(resolved_cells),
                "outcome_count": len(rows),
                "parity": {
                    "agreement_count": agreement_count,
                    "all_outcomes_agree": agreement_count == len(rows),
                },
                "evidence_status": "local_outcome_parity_review",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return rows_path, manifest_path


@dataclass(frozen=True)
class _RolloutSummary:
    success_rate: float
    efficiency_score: float


def _load_backend() -> tuple[Any, Any, Any]:
    try:
        gym = import_module("gymnasium")
        import_module("minigrid")
        numpy = import_module("numpy")
        ppo = import_module("stable_baselines3").PPO
    except ImportError as error:
        message = "Paper MiniGrid PPO evaluation requires `uv sync --extra minigrid`."
        raise RuntimeError(message) from error
    return gym, numpy, ppo


def _make_environment(
    gym: Any, numpy: Any, task: PaperMiniGridEnvironment, *, seed: int
) -> Any:
    environment = (
        _make_poscm_environment(task.poscm) if task.poscm else gym.make(task.env_id)
    )
    environment.reset(seed=seed)

    class NavigationActions(gym.ActionWrapper):  # type: ignore[misc]
        def __init__(self, wrapped: Any) -> None:
            super().__init__(wrapped)
            self.action_space = gym.spaces.Discrete(3)

        def action(self, action: int) -> int:
            return int(action)

    class SymbolicNavigation(gym.ObservationWrapper):  # type: ignore[misc]
        def __init__(self, wrapped: Any) -> None:
            super().__init__(wrapped)
            self.observation_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(14,), dtype=numpy.float32
            )

        def observation(self, observation: Any) -> Any:
            del observation
            unwrapped = self.unwrapped
            width = max(1, int(unwrapped.width) - 1)
            height = max(1, int(unwrapped.height) - 1)
            x, y = unwrapped.agent_pos
            direction = int(unwrapped.agent_dir)
            goal_x, goal_y = _first_goal_position(unwrapped)
            delta_x = float(goal_x - x) / float(width)
            delta_y = float(goal_y - y) / float(height)
            return numpy.asarray(
                (
                    float(x) / float(width),
                    float(y) / float(height),
                    *(1.0 if direction == index else 0.0 for index in range(4)),
                    float(goal_x) / float(width),
                    float(goal_y) / float(height),
                    delta_x,
                    delta_y,
                    abs(delta_x) + abs(delta_y),
                    1.0 if _front_blocked(unwrapped) else 0.0,
                    1.0 if (x, y) == (goal_x, goal_y) else 0.0,
                    float(unwrapped.step_count) / max(1.0, float(unwrapped.max_steps)),
                ),
                dtype=numpy.float32,
            )

    return SymbolicNavigation(NavigationActions(environment))


def _make_poscm_environment(task: PoscmMiniGridTask) -> Any:
    from minigrid.core.grid import Grid  # noqa: PLC0415
    from minigrid.core.mission import MissionSpace  # noqa: PLC0415
    from minigrid.core.world_object import Goal  # noqa: PLC0415
    from minigrid.minigrid_env import MiniGridEnv  # noqa: PLC0415

    required_actions = task.required_actions
    trap_prefixes = task.trap_prefixes
    width = max(task.size, sum(action == 2 for action in required_actions) + 3)

    class PoscmEnvironment(MiniGridEnv):
        def __init__(self) -> None:
            self.action_history: list[int] = []
            super().__init__(
                mission_space=MissionSpace(
                    mission_func=lambda: "follow the control signature"
                ),
                width=width,
                height=5,
                max_steps=max(16, width * 4),
                see_through_walls=True,
            )

        def _gen_grid(self, width: int, height: int) -> None:
            self.grid = Grid(width, height)
            self.grid.wall_rect(0, 0, width, height)
            self.agent_pos = (1, 2)
            self.agent_dir = 0
            self.put_obj(Goal(), width - 2, 2)
            self.mission = "follow the control signature"

        def reset(self, *args: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
            self.action_history = []
            return super().reset(*args, **kwargs)

        def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
            result: tuple[Any, Any, bool, bool, dict[str, Any]] = super().step(action)
            observation, reward, terminated, truncated, info = result
            self.action_history.append(int(action))
            history = tuple(self.action_history)
            trapped = any(
                len(history) >= len(prefix) and history[: len(prefix)] == prefix
                for prefix in trap_prefixes
            )
            required = (
                len(history) >= len(required_actions)
                and history[: len(required_actions)] == required_actions
            )
            if trapped:
                reward = min(0.0, float(reward))
                terminated = True
            elif required and self.step_count > task.reward_delay:
                reward = max(float(reward), 1.0)
                terminated = True
            elif required_actions and float(reward) > 0.0:
                reward = 0.0
                terminated = True
            info["success"] = bool(required and not trapped)
            return observation, float(reward), bool(terminated), bool(truncated), info

    return PoscmEnvironment()


def _ppo_kwargs(config: PaperMiniGridPpoConfig) -> dict[str, float | int]:
    return {
        "learning_rate": config.learning_rate,
        "n_steps": config.n_steps,
        "batch_size": config.batch_size,
        "n_epochs": config.n_epochs,
        "gamma": config.gamma,
        "gae_lambda": config.gae_lambda,
        "clip_range": config.clip_range,
        "ent_coef": config.ent_coef,
        "vf_coef": config.vf_coef,
    }


def _train_and_select_checkpoint(
    ppo: Any,
    model: Any,
    cell: PaperMiniGridCell,
    *,
    config: PaperMiniGridPpoConfig,
    gym: Any,
    numpy: Any,
    validation_seeds: tuple[int, ...] | None = None,
) -> Any:
    remaining = config.train_timesteps
    best_efficiency = -1.0
    selected_model = model
    with TemporaryDirectory(prefix="coo_paper_minigrid_ppo_") as directory:
        checkpoint = Path(directory) / "best_source_policy.zip"
        while remaining > 0:
            interval = min(config.checkpoint_interval, remaining)
            model.learn(
                total_timesteps=interval, reset_num_timesteps=False, progress_bar=False
            )
            remaining -= interval
            summary = _evaluate_policy(
                model,
                cell.source,
                seed=cell.source_seed,
                config=config,
                gym=gym,
                numpy=numpy,
                episodes=4,
                episode_seeds=validation_seeds,
            )
            if summary.efficiency_score >= best_efficiency:
                best_efficiency = summary.efficiency_score
                model.save(checkpoint)
        if checkpoint.is_file():
            selected_model = ppo.load(checkpoint)
    return selected_model


def _evaluate_policy(
    model: Any,
    task: PaperMiniGridEnvironment,
    *,
    seed: int,
    config: PaperMiniGridPpoConfig,
    gym: Any,
    numpy: Any,
    episodes: int | None = None,
    episode_seeds: tuple[int, ...] | None = None,
) -> _RolloutSummary:
    successes: list[float] = []
    efficiencies: list[float] = []
    resolved_seeds = episode_seeds or (seed,) * (episodes or config.evaluation_episodes)
    for episode_seed in resolved_seeds:
        environment = _make_environment(gym, numpy, task, seed=episode_seed)
        try:
            observation, _ = environment.reset(seed=episode_seed)
            oracle_steps = _oracle_steps(environment.unwrapped, task.poscm)
            for step in range(1, int(environment.unwrapped.max_steps) + 1):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = environment.step(
                    int(action)
                )
                if terminated or truncated:
                    success = bool(float(reward) > 0.0 or info.get("success"))
                    successes.append(float(success))
                    efficiencies.append(_efficiency(success, step, oracle_steps))
                    break
        finally:
            environment.close()
    return _RolloutSummary(
        sum(successes) / len(successes), sum(efficiencies) / len(efficiencies)
    )


def _evaluate_random(
    task: PaperMiniGridEnvironment,
    *,
    seed: int,
    config: PaperMiniGridPpoConfig,
    gym: Any,
    numpy: Any,
    environment_seeds: tuple[int, ...] | None = None,
    action_seeds: tuple[int, ...] | None = None,
) -> _RolloutSummary:
    successes: list[float] = []
    efficiencies: list[float] = []
    resolved_environment_seeds = environment_seeds or (seed,) * (
        config.evaluation_episodes
    )
    resolved_action_seeds = action_seeds or (seed,) * len(resolved_environment_seeds)
    if len(resolved_environment_seeds) != len(resolved_action_seeds):
        _raise_invalid("random environment and action seeds must align")
    for environment_seed, action_seed in zip(
        resolved_environment_seeds, resolved_action_seeds, strict=True
    ):
        environment = _make_environment(gym, numpy, task, seed=environment_seed)
        try:
            observation, _ = environment.reset(seed=environment_seed)
            del observation
            generator = numpy.random.default_rng(action_seed)
            oracle_steps = _oracle_steps(environment.unwrapped, task.poscm)
            for step in range(1, int(environment.unwrapped.max_steps) + 1):
                _, reward, terminated, truncated, info = environment.step(
                    int(generator.integers(environment.action_space.n))
                )
                if terminated or truncated:
                    success = bool(float(reward) > 0.0 or info.get("success"))
                    successes.append(float(success))
                    efficiencies.append(_efficiency(success, step, oracle_steps))
                    break
        finally:
            environment.close()
    return _RolloutSummary(
        sum(successes) / len(successes), sum(efficiencies) / len(efficiencies)
    )


def _first_goal_position(environment: Any) -> tuple[int, int]:
    for x in range(int(environment.width)):
        for y in range(int(environment.height)):
            if getattr(environment.grid.get(x, y), "type", "") == "goal":
                return x, y
    return int(environment.width) - 1, int(environment.height) - 1


def _front_blocked(environment: Any) -> bool:
    x, y = environment.agent_pos
    direction_vectors = ((1, 0), (0, 1), (-1, 0), (0, -1))
    dx, dy = direction_vectors[int(environment.agent_dir) % 4]
    cell = environment.grid.get(int(x) + dx, int(y) + dy)
    return bool(
        cell is not None
        and (
            getattr(cell, "type", "") in {"wall", "lava"}
            or (
                getattr(cell, "type", "") == "door"
                and not getattr(cell, "is_open", False)
            )
        )
    )


def _oracle_steps(environment: Any, poscm: PoscmMiniGridTask | None) -> int:
    if poscm is not None:
        return len(poscm.required_actions) + poscm.reward_delay
    x, y = environment.agent_pos
    goal_x, goal_y = _first_goal_position(environment)
    distance = abs(goal_x - int(x)) + abs(goal_y - int(y))
    return max(1, distance + 1)


def _efficiency(success: bool, steps: int, oracle_steps: int) -> float:
    if not success:
        return 0.0
    return max(0.0, min(1.0, float(oracle_steps) / float(max(1, steps))))


def _normalized_transfer_effect(zero_shot: float, random: float) -> float:
    denominator = 1.0 - random
    if denominator <= 1e-9:
        return 0.0
    return max(-1.0, min(1.0, (zero_shot - random) / denominator))


__all__ = [
    "PaperMiniGridCell",
    "PaperMiniGridEnvironment",
    "PaperMiniGridIndependentSeeds",
    "PaperMiniGridOutcome",
    "PaperMiniGridPpoConfig",
    "PoscmMiniGridTask",
    "evaluate_paper_minigrid_cell",
    "evaluate_paper_minigrid_cell_independent",
    "paper_minigrid_cells",
    "write_paper_minigrid_policy_evaluation",
]
