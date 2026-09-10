from __future__ import annotations

import pytest

from causal_observation_reproduction.reference.quotient_family import (
    CausalQuotientFamilySpec,
    generate_causal_quotient_family,
)


@pytest.mark.parametrize("regime", ["positive", "null", "invalid"])
def test_controlled_quotient_regimes_are_oracle_checked_and_scorer_separated(
    regime: str,
) -> None:
    family = generate_causal_quotient_family(
        spec=CausalQuotientFamilySpec(
            regime=regime,  # type: ignore[arg-type]
            horizon=3,
            reward_delay=1,
            continuation_dependency_depth=2,
            distractor_count=2,
            seed=29,
        )
    )
    sidecar = family.scorer_sidecar
    resolved = next(iter(sidecar["world_reference_quotients"].values()))

    assert resolved["declared_regime"] == regime
    assert resolved["resolved_regime"] == regime
    assert sidecar["artifact_scope"] == "scorer_only_oracle"
    assert sidecar["deployment_visible"] is False
    assert family.deployment_manifest["artifact_scope"] == "deployment_protocol"
    assert "reference_quotient" not in family.deployment_manifest
    assert "reference_quotient" not in family.semantic_world
    assert "scorer_only_reference_quotient" not in family.semantic_world
    labels = resolved["reference_quotient"]["labels_by_context"]
    if regime == "null":
        assert len(set(labels.values())) == 1
    else:
        assert len(set(labels.values())) == 2


def test_family_parameters_are_preserved_without_turning_oracle_labels_into_features() -> (
    None
):
    family = generate_causal_quotient_family(
        spec=CausalQuotientFamilySpec(
            regime="positive",
            horizon=4,
            reward_delay=2,
            continuation_dependency_depth=2,
            distractor_count=5,
            probe_cost=0.4,
            probe_reliability=0.7,
            sensing_budget=3,
            alias_persistence=2,
            seed=31,
        )
    )
    protocol = family.deployment_manifest["protocol"]

    assert protocol["probe_cost"] == 0.4
    assert protocol["probe_reliability"] == 0.7
    assert protocol["sensing_budget"] == 3
    assert protocol["alias_persistence"] == 2
    assert (
        family.semantic_world["reward_model"]["contextual_target_scope"]
        == "controlled_causal_quotient_family"
    )


def test_family_spec_fails_closed_for_invalid_control_parameters() -> None:
    with pytest.raises(ValueError, match="reliability"):
        CausalQuotientFamilySpec(regime="positive", probe_reliability=1.1)
    with pytest.raises(ValueError, match="smaller than horizon"):
        CausalQuotientFamilySpec(regime="positive", horizon=2, reward_delay=2)
