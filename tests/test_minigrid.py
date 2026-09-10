from __future__ import annotations

import json
from typing import TYPE_CHECKING

from causal_observation_reproduction.experiments.minigrid import (
    MiniGridPairMatrixConfig,
    generate_minigrid_pair_matrix,
    structural_readiness,
    write_minigrid_design,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_publication_pair_matrix_matches_the_source_derived_default_design() -> None:
    pairs = generate_minigrid_pair_matrix()

    assert len(pairs) == 4
    assert {pair.perturbation_type for pair in pairs} == {
        "same_env_same_seed",
        "same_family_scale_change",
    }
    assert {
        (pair.source_env_id, pair.target_env_id, pair.source_seed, pair.target_seed)
        for pair in pairs
    } == {
        ("MiniGrid-Empty-5x5-v0", "MiniGrid-Empty-5x5-v0", 17, 17),
        ("MiniGrid-Empty-5x5-v0", "MiniGrid-Empty-8x8-v0", 17, 17),
        ("MiniGrid-Empty-8x8-v0", "MiniGrid-Empty-5x5-v0", 17, 17),
        ("MiniGrid-Empty-8x8-v0", "MiniGrid-Empty-8x8-v0", 17, 17),
    }
    assert len({pair.pair_id for pair in pairs}) == len(pairs)


def test_identity_pair_has_unit_structural_readiness() -> None:
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

    readiness = structural_readiness(pair)
    assert readiness.edge_jaccard_similarity == 1.0
    assert readiness.state_count_ratio == 1.0
    assert readiness.score == 1.0
    assert readiness.band == "high"


def test_design_writer_preserves_the_optional_policy_boundary(tmp_path: Path) -> None:
    rows_path, manifest_path = write_minigrid_design(tmp_path)

    assert len(rows_path.read_text(encoding="utf-8").splitlines()) == 4
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["policy_evaluation_status"] == "requires_optional_policy_evaluator"
