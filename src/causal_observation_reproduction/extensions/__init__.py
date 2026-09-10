"""Explicit boundary for experiments that are not evidence in the manuscript."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def run_certified_cache_control(output_dir: str | Path) -> tuple[Path, ...]:
    """Run the post-manuscript certified cache-control application."""

    from causal_observation_reproduction.certified_cache_control import (
        write_cache_control_dataset,
    )

    return tuple(write_cache_control_dataset(output_dir).values())


def run_sequential_robustness(output_dir: str | Path) -> tuple[Path, ...]:
    """Run the post hoc history-dependent sequential robustness design."""

    from causal_observation_reproduction.experiments.sequential_robustness import (
        write_sequential_robustness,
    )

    return write_sequential_robustness(output_dir)


def run_one_step_strengthened(output_dir: str | Path) -> tuple[Path, ...]:
    """Run the larger independent-test one-step robustness design."""

    from causal_observation_reproduction.experiments.one_step_learning import (
        write_one_step_calibration,
    )

    return write_one_step_calibration(output_dir)


def run_poscm_transfer(output_dir: str | Path) -> tuple[Path, ...]:
    """Run the additional POSCM transfer design."""

    from causal_observation_reproduction.experiments.poscm_transfer import (
        write_poscm_transfer_design,
    )

    return write_poscm_transfer_design(output_dir)


def run_poscm_dqn(output_dir: str | Path) -> tuple[Path, ...]:
    """Run the additional local-DQN policy evaluation."""

    from causal_observation_reproduction.experiments.poscm_policy import (
        write_poscm_dqn_evaluation,
    )

    return write_poscm_dqn_evaluation(output_dir)


def run_minigrid_design(output_dir: str | Path) -> tuple[Path, ...]:
    """Run the additional synthetic MiniGrid design."""

    from causal_observation_reproduction.experiments.minigrid import (
        write_minigrid_design,
    )

    return write_minigrid_design(output_dir)


def run_minigrid_policy(output_dir: str | Path) -> tuple[Path, ...]:
    """Run the additional local-PPO policy evaluation."""

    from causal_observation_reproduction.experiments.paper_minigrid_policy import (
        write_paper_minigrid_policy_evaluation,
    )

    return write_paper_minigrid_policy_evaluation(output_dir)


def run_minigrid_robustness(output_dir: str | Path) -> tuple[Path, ...]:
    """Run the additional independent-seed PPO robustness experiment."""

    from causal_observation_reproduction.experiments.paper_minigrid_robustness import (
        write_paper_minigrid_robustness,
    )

    return write_paper_minigrid_robustness(output_dir)


__all__ = [
    "run_certified_cache_control",
    "run_minigrid_design",
    "run_minigrid_policy",
    "run_minigrid_robustness",
    "run_one_step_strengthened",
    "run_poscm_dqn",
    "run_poscm_transfer",
    "run_sequential_robustness",
]
