from __future__ import annotations

import pytest

from causal_observation_reproduction.reference.interventional_signature import (
    InterventionalActionSignature,
    ReferenceQuotient,
    build_generated_poscm_reference_quotient_sidecar,
    classify_declared_quotient_regime,
    derive_finite_horizon_action_signatures,
    enumerate_binary_contexts,
)
from causal_observation_reproduction.reference.topology_factory import (
    PoscmTopologyFactoryConfig,
    generate_poscm_topology,
)


def test_semantic_world_signature_is_scorer_only_and_bounded() -> None:
    world = generate_poscm_topology(
        config=PoscmTopologyFactoryConfig(
            node_count=6, action_count=2, horizon=2, seed=17
        ),
        index=0,
    ).semantic_world
    latent = list(world["observation_model"]["latent_variables"])
    contexts = enumerate_binary_contexts(world, variable_ids=latent[:1], max_contexts=2)

    quotient = derive_finite_horizon_action_signatures(
        world,
        contexts=contexts,
        horizon=2,
        scorer_seed=17,
    )
    payload = quotient.scorer_payload()

    assert payload["artifact_scope"] == "scorer_only_oracle"
    assert payload["selection_eligible"] is False
    assert payload["deployment_visible"] is False
    assert set(payload["labels_by_context"]) == set(contexts)
    assert all("state_assignment" not in row for row in payload["signatures"])


def test_declared_regimes_are_checked_against_reference_signatures() -> None:
    reference = ReferenceQuotient(
        signatures=(
            InterventionalActionSignature("a", 3, (0.0, 1.0)),
            InterventionalActionSignature("b", 3, (0.0, 1.0)),
            InterventionalActionSignature("c", 3, (1.0, 0.0)),
        ),
        scorer_seed=0,
    )
    positive = classify_declared_quotient_regime(
        declared_manifest={
            "declared_regime": "positive",
            "proposed_observation_classes": {"a": "left", "b": "left", "c": "right"},
        },
        reference_quotient=reference,
    )
    invalid = classify_declared_quotient_regime(
        declared_manifest={
            "declared_regime": "invalid",
            "proposed_observation_classes": {
                "a": "merged",
                "b": "merged",
                "c": "merged",
            },
        },
        reference_quotient=reference,
    )

    assert positive.resolved_regime == "positive"
    assert invalid.resolved_regime == "invalid"
    with pytest.raises(ValueError, match="disagrees"):
        classify_declared_quotient_regime(
            declared_manifest={
                "declared_regime": "positive",
                "proposed_observation_classes": {
                    "a": "merged",
                    "b": "merged",
                    "c": "merged",
                },
            },
            reference_quotient=reference,
        )


def test_null_regime_and_context_bound_fail_closed() -> None:
    reference = ReferenceQuotient(
        signatures=(
            InterventionalActionSignature("a", 2, (0.0, 0.0)),
            InterventionalActionSignature("b", 2, (0.0, 0.0)),
        ),
        scorer_seed=0,
    )
    resolved = classify_declared_quotient_regime(
        declared_manifest={
            "declared_regime": "null",
            "proposed_observation_classes": {"a": "same", "b": "same"},
        },
        reference_quotient=reference,
    )
    assert resolved.resolved_regime == "null"

    with pytest.raises(ValueError, match="exceeds"):
        enumerate_binary_contexts(
            {"state_variables": [{"variable_id": "x"}, {"variable_id": "y"}]},
            max_contexts=2,
        )


def test_generated_bridge_reference_labels_stay_in_scorer_sidecar() -> None:
    world = generate_poscm_topology(
        config=PoscmTopologyFactoryConfig(
            node_count=6, action_count=2, horizon=2, seed=23
        ),
        index=0,
    ).semantic_world
    latent = list(world["observation_model"]["latent_variables"])
    contexts = enumerate_binary_contexts(world, variable_ids=latent[:1], max_contexts=2)
    bridge = {
        "bridge_id": "bridge-1",
        "worlds": [{"world_id": "world-1", "semantic_world": world}],
    }
    reference = derive_finite_horizon_action_signatures(
        world, contexts=contexts, horizon=2, scorer_seed=23
    )
    proposed_classes = reference.labels_by_context

    sidecar = build_generated_poscm_reference_quotient_sidecar(
        bridge,
        contexts_by_world={"world-1": contexts},
        manifests_by_world={
            "world-1": {
                "declared_regime": "null"
                if len(set(proposed_classes.values())) == 1
                else "positive",
                "proposed_observation_classes": proposed_classes,
            }
        },
        horizon=2,
        scorer_seed=23,
    )

    assert sidecar["artifact_scope"] == "scorer_only_oracle"
    assert sidecar["deployment_visible"] is False
    assert "scorer_only_reference_quotient" not in bridge["worlds"][0]
