from __future__ import annotations

from causal_observation_reproduction.reference.artifacts import assert_json_roundtrip
from causal_observation_reproduction.reference.scm_complexity import (
    build_structural_equation_spec,
)
from causal_observation_reproduction.reference.semantic_world import (
    ActionSpaceSpec,
    InterventionSpec,
    ObservationModelSpec,
    RewardModelSpec,
    SemanticWorldSpec,
    StateVariableSpec,
    TransitionModelRowSpec,
    validate_semantic_world_payload,
)


def test_semantic_world_spec_serializes_to_existing_payload_contract() -> None:
    world = _semantic_world(index=0)
    payload = world.as_payload()

    assert assert_json_roundtrip(payload) == payload
    assert payload["schema_version"] == "beb.generated_poscm_semantic_world.v0"
    assert payload["semantic_world_id"].startswith("generated-poscm-semantic-world-")
    assert payload["action_space"]["action_effect_matrix"] == [[1.0]]
    assert (
        payload["transition_model"][0]["structural_equation"]["variable_id"]
        == "outcome"
    )
    assert payload["interventions"][0]["targets"] == [["state", "outcome"]]
    assert all(validate_semantic_world_payload(payload).values())


def test_semantic_world_id_is_independent_of_occurrence_index() -> None:
    assert (
        _semantic_world(index=0).semantic_world_id
        == _semantic_world(index=7).semantic_world_id
    )


def _semantic_world(*, index: int) -> SemanticWorldSpec:
    structural_equation = build_structural_equation_spec(
        variable_id="outcome",
        parent_variables=("state",),
        action_inputs=("action_0",),
    )
    return SemanticWorldSpec(
        state_variables=(
            StateVariableSpec(variable_id="state", role="context", observed=True),
            StateVariableSpec(variable_id="outcome", role="outcome", observed=True),
            StateVariableSpec(variable_id="latent", role="noise", observed=False),
        ),
        action_space=ActionSpaceSpec(
            action_variables=("action_0",), branching_factor=1
        ),
        transition_model=(
            TransitionModelRowSpec(
                variable_id="outcome",
                parents=("state",),
                action_inputs=("action_0",),
                latent_noise_inputs=(),
                equation="parent_action_noisy_or",
                structural_equation=structural_equation,
                transition_delay=1,
            ),
        ),
        reward_model=RewardModelSpec(
            reward_id="semantic_reward",
            outcome_variables=("outcome",),
            action_variables=("action_0",),
            reward_delay=1,
            success_reward=1.0,
            failure_reward=-0.05,
            success_condition="outcome reaches terminal state",
            credit_assignment_target="attribute_reward_to_boundary_mediator",
        ),
        observation_model=ObservationModelSpec(
            observed_variables=("state", "outcome"),
            latent_variables=("latent",),
            observation_noise=0.1,
            proxy_reliability=0.75,
            spurious_correlation_strength=0.2,
            event_emitters={"observable_endpoint_event": ["outcome"]},
        ),
        interventions=(
            InterventionSpec(
                intervention_id="perturb_edge_strength",
                intervention_type="edge_strength_shift",
                targets=(("state", "outcome"),),
                parameters={"delta": 0.3},
            ),
        ),
        difficulty_vector={
            "horizon": 8.0,
            "branching_factor": 1.0,
            "action_count": 1.0,
            "reward_delay": 1.0,
            "observation_noise": 0.1,
            "latent_confounding_strength": 0.3,
            "proxy_reliability": 0.75,
            "spurious_correlation_strength": 0.2,
            "topology_edge_density": 0.5,
            "latent_fraction": 0.333333,
            "complexity_tier": "tier_0_linear",
            "interaction_order": 1.0,
            "nonlinear_transform_count": 0.0,
            "temporal_lag": 0.0,
        },
        complexity={
            "complexity_tier": "tier_0_linear",
            "complexity_vector": {"polynomial_order": 1.0},
            "graph_metrics": {"node_count": 3},
            "equation_summary": {"equation_count": 1},
        },
        target_region="positive_boundary_advantage",
        index=index,
    )
