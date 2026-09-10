from __future__ import annotations

import json

import pytest

from causal_observation_reproduction.experiments.paper_minigrid_policy import (
    PaperMiniGridIndependentSeeds,
    PaperMiniGridPpoConfig,
    paper_minigrid_cells,
)
from causal_observation_reproduction.experiments.paper_minigrid_robustness import (
    PaperMiniGridRobustnessConfig,
    write_paper_minigrid_robustness,
)


def test_independent_seed_plan_rejects_misaligned_random_panel() -> None:
    with pytest.raises(ValueError, match="align"):
        PaperMiniGridIndependentSeeds(
            optimizer_seed=1,
            validation_seeds=(2,),
            source_evaluation_seeds=(3,),
            target_evaluation_seeds=(4, 5),
            random_action_seeds=(6,),
        )


def test_robustness_config_rejects_overlapping_seed_panels() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        PaperMiniGridRobustnessConfig(
            optimizer_seeds=(1,),
            validation_seeds=(1,),
            source_evaluation_seeds=(2,),
            target_evaluation_seeds=(3,),
            random_action_seeds=(4,),
        )


def test_optional_robustness_writer_uses_training_runs_as_replication_units(
    tmp_path,
) -> None:
    pytest.importorskip("gymnasium")
    pytest.importorskip("minigrid")
    pytest.importorskip("stable_baselines3")
    config = PaperMiniGridRobustnessConfig(
        ppo=PaperMiniGridPpoConfig(
            train_timesteps=64,
            checkpoint_interval=64,
            evaluation_episodes=1,
            repeats=1,
            n_steps=32,
            batch_size=16,
            n_epochs=1,
        ),
        optimizer_seeds=(101, 211),
        validation_seeds=(6_001,),
        source_evaluation_seeds=(7_001, 7_002),
        target_evaluation_seeds=(8_001, 8_002),
        random_action_seeds=(9_001, 9_002),
    )

    runs_path, summaries_path, manifest_path = write_paper_minigrid_robustness(
        tmp_path, config, cells=(paper_minigrid_cells()[4],)
    )

    assert len(runs_path.read_text(encoding="utf-8").splitlines()) == 2
    assert len(summaries_path.read_text(encoding="utf-8").splitlines()) == 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["replication_unit"] == "independently_seeded_ppo_training_run"
    assert manifest["training_run_count"] == 2
    assert manifest["evaluation_episode_count"] == 12
    assert manifest["historical_protocol_modified"] is False
    assert manifest["paper_level_conclusion"] is False
