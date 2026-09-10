from __future__ import annotations

from causal_observation_reproduction.reference.domain import (
    CausalGraphSpec,
    EdgeSpec,
    NodeSpec,
    ResponseRegion,
    TopologySpec,
)


def test_causal_graph_spec_derives_variable_sets() -> None:
    graph = CausalGraphSpec(
        nodes=(
            NodeSpec("action", "action", True),
            NodeSpec("latent_boundary", "mediator", False, ("latent_boundary_event",)),
            NodeSpec("outcome", "outcome", True, ("observable_endpoint_event",)),
        ),
        edges=(
            EdgeSpec("action", "latent_boundary"),
            EdgeSpec("latent_boundary", "outcome"),
        ),
    )

    assert graph.hidden_edges == (
        ("action", "latent_boundary"),
        ("latent_boundary", "outcome"),
    )
    assert graph.observed_variables == ("action", "outcome")
    assert graph.latent_variables == ("latent_boundary",)
    assert graph.action_variables == ("action",)
    assert graph.outcome_variables == ("outcome",)
    assert graph.event_trace_variables("latent_boundary_event") == ("latent_boundary",)


def test_topology_spec_keeps_legacy_evidence_region_projection() -> None:
    topology = TopologySpec(
        motif_id="observability_gap",
        world_topology="graph_observability_gap",
        sweep_variant_id="observability_gap",
        expected_response_region=ResponseRegion.OBSERVABILITY_SCOPE_QUALIFIER,
        graph=CausalGraphSpec(
            nodes=(NodeSpec("boundary", "mediator", True),),
            edges=(),
        ),
        topology_tags=("observability_gap",),
        hypothesis_targets=("H_BEB_OBSERVABILITY_SCOPE_CONDITION",),
        metric_expectations={},
        generator_params={},
    )

    assert topology.variables[0].variable_id == "boundary"
    assert topology.expected_evidence_region == "observability_scope_qualifier"
