from __future__ import annotations

import pytest

from causal_observation_reproduction.reference.sequential_contract import (
    POLICY_VISIBLE_CHANNELS,
    FeedbackBoundary,
    assert_complete_feedback_boundary,
    build_smoke_block,
    smoke_contract,
)


def test_paired_block_gives_every_information_arm_the_same_latent_world() -> None:
    block = build_smoke_block(11)

    assert block.has_matched_latent_worlds()
    assert len(set(block.latent_world_ids_by_arm().values())) == 1
    assert "bounded_generic_voi" in block.arms
    assert "causal_quotient" in block.arms


def test_refinement_has_positive_delayed_value_without_immediate_reward() -> None:
    block = build_smoke_block(3)

    assert block.episode.immediate_information_reward == 0.0
    assert block.episode.delayed_net_gain() > 0.0
    assert block.episode.validates_delayed_value()


def test_complete_observation_boundary_rejects_telemetry_only_alias() -> None:
    left = FeedbackBoundary("same", 0.0, False, "ok", "", "frame", "none")
    right = FeedbackBoundary("same", 1.0, False, "ok", "", "frame", "none")

    with pytest.raises(ValueError, match="telemetry alias"):
        assert_complete_feedback_boundary(left, right)

    assert left.project(("telemetry",)) == right.project(("telemetry",))
    assert left.project(POLICY_VISIBLE_CHANNELS) != right.project(
        POLICY_VISIBLE_CHANNELS
    )


def test_smoke_payload_records_all_three_validity_guards() -> None:
    payload = smoke_contract(5)

    assert payload["matched_latent_worlds"] is True
    assert payload["delayed_value_supported"] is True
    assert payload["feedback_leakage_detected"] is True
