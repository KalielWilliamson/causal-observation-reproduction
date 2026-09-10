from __future__ import annotations

import json
from typing import TYPE_CHECKING

from causal_observation_reproduction.experiments.poscm_transfer import (
    evaluate_poscm_transfer_pair,
    generate_poscm_transfer_pairs,
    poscm_transfer_readiness,
    write_poscm_transfer_design,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_poscm_transfer_design_retains_each_declared_pair_class() -> None:
    pairs = generate_poscm_transfer_pairs()

    assert len(pairs) == 4
    assert {pair.pair_class for pair in pairs} == {
        "same_motif_parameter_perturbation",
        "same_action_deeper_gate",
        "branching_change_same_motif_family",
        "graph_close_control_break",
    }


def test_poscm_transfer_default_cells_match_the_source_derived_factors() -> None:
    same_motif = generate_poscm_transfer_pairs()[0]

    assert same_motif.source.action_count == 2
    assert same_motif.source.gate_depth == 3
    assert same_motif.source.distractor_count == 1
    assert same_motif.source.seed == 17
    assert same_motif.target.action_count == 2
    assert same_motif.target.gate_depth == 3
    assert same_motif.target.distractor_count == 1
    assert same_motif.target.seed == 23


def test_control_break_is_structurally_close_but_readiness_penalized() -> None:
    pair = next(
        pair
        for pair in generate_poscm_transfer_pairs()
        if pair.pair_class == "graph_close_control_break"
    )

    readiness = poscm_transfer_readiness(pair)
    assert readiness.control_compatibility == 0.0
    assert readiness.structural_compatibility == 1.0
    assert readiness.negative_control_risk == 1.0
    assert readiness.decision_band == "treat_as_new_domain"


def test_poscm_transfer_runner_writes_local_policy_outcomes(tmp_path: Path) -> None:
    rows_path, manifest_path = write_poscm_transfer_design(tmp_path)

    assert len(rows_path.read_text(encoding="utf-8").splitlines()) == 4
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["policy_backend"] == "tabular_q_learning"


def test_portable_policy_trains_on_the_declared_source_world() -> None:
    pair = generate_poscm_transfer_pairs()[0]

    result = evaluate_poscm_transfer_pair(pair)
    assert result.source_mastered is True
    assert result.target_random_success_rate == 0.125
    assert result.policy_backend == "tabular_q_learning"
