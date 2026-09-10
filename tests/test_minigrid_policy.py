from __future__ import annotations

import json

import pytest

from causal_observation_reproduction.experiments.minigrid import (
    MiniGridPairMatrixConfig,
    generate_minigrid_pair_matrix,
)
from causal_observation_reproduction.experiments.minigrid_policy import (
    MiniGridEvaluationConfig,
    MiniGridPolicyConfig,
    evaluate_minigrid_pair,
    write_minigrid_policy_evaluation,
)
from causal_observation_reproduction.experiments.paper_minigrid_policy import (
    PaperMiniGridPpoConfig,
    evaluate_paper_minigrid_cell,
    paper_minigrid_cells,
)


def test_minigrid_policy_config_rejects_empty_protocol() -> None:
    with pytest.raises(ValueError, match="positive"):
        MiniGridPolicyConfig(train_timesteps=0)


def test_paper_minigrid_panel_is_frozen_and_balanced() -> None:
    cells = paper_minigrid_cells()

    assert len(cells) == 8
    assert sum(int(cell.expected_transfer) for cell in cells) == 5
    assert {cell.control_role for cell in cells} == {
        "positive_control",
        "near_neighbor",
        "negative_control",
    }


def test_optional_paper_ppo_backend_evaluates_a_declared_control() -> None:
    pytest.importorskip("gymnasium")
    pytest.importorskip("minigrid")
    pytest.importorskip("stable_baselines3")

    result = evaluate_paper_minigrid_cell(
        paper_minigrid_cells()[4],
        PaperMiniGridPpoConfig(
            train_timesteps=64,
            checkpoint_interval=64,
            evaluation_episodes=1,
            repeats=1,
            n_steps=32,
            batch_size=16,
            n_epochs=1,
        ),
    )

    assert result.policy_backend == "stable_baselines3_ppo"


def test_optional_dqn_backend_smoke_evaluates_a_declared_identity_pair() -> None:
    pytest.importorskip("gymnasium")
    pytest.importorskip("minigrid")
    pytest.importorskip("stable_baselines3")
    pair = generate_minigrid_pair_matrix(
        MiniGridPairMatrixConfig(
            families=("Empty",),
            sizes=(5,),
            seeds=(17,),
            include_scale=False,
            include_seed_shift=False,
            include_causal_negative_controls=False,
        )
    )[0]

    result = evaluate_minigrid_pair(
        pair,
        MiniGridPolicyConfig(
            train_timesteps=64, evaluation_episodes=1, learning_starts=1
        ),
    )
    assert result.policy_backend == "stable_baselines3_dqn"


def test_optional_dqn_backend_handles_a_declared_scale_pair() -> None:
    pytest.importorskip("gymnasium")
    pytest.importorskip("minigrid")
    pytest.importorskip("stable_baselines3")
    pair = next(
        pair
        for pair in generate_minigrid_pair_matrix(
            MiniGridPairMatrixConfig(
                families=("Empty",),
                sizes=(5, 8),
                seeds=(17,),
                include_same=False,
                include_seed_shift=False,
                include_causal_negative_controls=False,
            )
        )
        if pair.perturbation_type == "same_family_scale_change"
    )

    result = evaluate_minigrid_pair(
        pair,
        MiniGridPolicyConfig(
            train_timesteps=64, evaluation_episodes=1, learning_starts=1
        ),
    )

    assert result.policy_backend == "stable_baselines3_dqn"


def test_optional_dqn_writer_keeps_pair_design_and_outcomes_together(tmp_path) -> None:
    pytest.importorskip("gymnasium")
    pytest.importorskip("minigrid")
    pytest.importorskip("stable_baselines3")
    matrix = MiniGridPairMatrixConfig(
        families=("Empty",),
        sizes=(5,),
        seeds=(17,),
        include_scale=False,
        include_seed_shift=False,
        include_causal_negative_controls=False,
    )

    rows_path, manifest_path = write_minigrid_policy_evaluation(
        tmp_path,
        MiniGridEvaluationConfig(
            pair_matrix=matrix,
            policy=MiniGridPolicyConfig(
                train_timesteps=64, evaluation_episodes=1, learning_starts=1
            ),
        ),
    )

    assert len(rows_path.read_text(encoding="utf-8").splitlines()) == 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_mastery"]["source_mastered_pair_count"] in {0, 1}
