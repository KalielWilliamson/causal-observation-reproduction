from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from causal_observation_reproduction.experiments.poscm_policy import (
    PoscmDqnConfig,
    evaluate_poscm_dqn_pair,
    write_poscm_dqn_evaluation,
)
from causal_observation_reproduction.experiments.poscm_transfer import (
    PoscmTransferConfig,
    generate_poscm_transfer_pairs,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_poscm_dqn_config_rejects_an_empty_protocol() -> None:
    with pytest.raises(ValueError, match="positive"):
        PoscmDqnConfig(train_timesteps=0)


def test_optional_poscm_dqn_backend_smoke_evaluates_a_declared_pair() -> None:
    pytest.importorskip("gymnasium")
    pytest.importorskip("stable_baselines3")

    result = evaluate_poscm_dqn_pair(
        generate_poscm_transfer_pairs()[0],
        PoscmDqnConfig(train_timesteps=64, evaluation_episodes=1, learning_starts=1),
    )

    assert result.policy_backend == "stable_baselines3_dqn"


def test_optional_poscm_dqn_handles_a_pair_with_a_deeper_target() -> None:
    pytest.importorskip("gymnasium")
    pytest.importorskip("stable_baselines3")

    pair = next(
        pair
        for pair in generate_poscm_transfer_pairs()
        if pair.pair_class == "same_action_deeper_gate"
    )
    result = evaluate_poscm_dqn_pair(
        pair,
        PoscmDqnConfig(train_timesteps=64, evaluation_episodes=1, learning_starts=1),
    )

    assert result.policy_backend == "stable_baselines3_dqn"


def test_optional_poscm_dqn_writer_keeps_the_fixed_pair_cell(tmp_path: Path) -> None:
    pytest.importorskip("gymnasium")
    pytest.importorskip("stable_baselines3")

    rows_path, manifest_path = write_poscm_dqn_evaluation(
        tmp_path,
        PoscmDqnConfig(train_timesteps=64, evaluation_episodes=1, learning_starts=1),
        transfer_config=PoscmTransferConfig(pair_limit=1, evaluation_episodes=1),
    )

    assert len(rows_path.read_text(encoding="utf-8").splitlines()) == 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["pair_count"] == 1
    assert manifest["source_mastery"]["source_mastered_pair_count"] in {0, 1}
