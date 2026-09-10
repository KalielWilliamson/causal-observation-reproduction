from __future__ import annotations

from typing import Any

from causal_observation_reproduction.reference.domain import (
    CausalGraphSpec,
    EdgeSpec,
    NodeSpec,
    ResponseRegion,
    TopologySpec,
)

GRAPH_DIFFERENTIAL_METRIC_MATRIX_WORLD_TOPOLOGY = "graph_differential_metric_matrix"
GRAPH_ENDPOINT_DIRECT_CAUSE_WORLD_TOPOLOGY = "graph_endpoint_direct_cause"
GRAPH_COLLIDER_CONFOUNDED_WORLD_TOPOLOGY = "graph_collider_confounded"
GRAPH_OBSERVABILITY_GAP_WORLD_TOPOLOGY = "graph_observability_gap"

BOUNDARY_MEDIATOR_MOTIF = "boundary_mediator"
ENDPOINT_DIRECT_MOTIF = "endpoint_direct"
COLLIDER_CONFOUNDED_MOTIF = "collider_confounded"
OBSERVABILITY_GAP_MOTIF = "observability_gap"


CausalObservabilityVariable = NodeSpec
CausalObservabilityTopologySpec = TopologySpec


def causal_observability_topology_specs() -> dict[str, CausalObservabilityTopologySpec]:
    return {
        BOUNDARY_MEDIATOR_MOTIF: CausalObservabilityTopologySpec(
            motif_id=BOUNDARY_MEDIATOR_MOTIF,
            world_topology=GRAPH_DIFFERENTIAL_METRIC_MATRIX_WORLD_TOPOLOGY,
            sweep_variant_id="differential_metric_matrix",
            expected_response_region=ResponseRegion.POSITIVE_BOUNDARY_ADVANTAGE,
            graph=CausalGraphSpec(
                nodes=(
                    CausalObservabilityVariable("context", "context", True),
                    CausalObservabilityVariable(
                        "boundary_root",
                        "mediator",
                        True,
                        ("observable_boundary_event",),
                    ),
                    CausalObservabilityVariable(
                        "endpoint_outcome",
                        "outcome",
                        True,
                        ("observable_endpoint_event",),
                    ),
                ),
                edges=(
                    EdgeSpec("context", "boundary_root"),
                    EdgeSpec("boundary_root", "endpoint_outcome"),
                ),
            ),
            topology_tags=(
                "boundary_mediator",
                "delayed_credit",
                "endpoint_root_separation",
            ),
            hypothesis_targets=(
                "H_BEB_IDENTIFIABLE_BOUNDARY_MEDIATOR_REGION",
                "H_BEB_BOUNDARY_EVENTIZATION_IMPROVES_ROOT_ATTRIBUTION",
            ),
            metric_expectations={
                "root_cause_mrr": "boundary_complete_gt_trajectory",
                "divergence_localization_accuracy": "trajectory_can_win",
            },
            generator_params={"mediator_depth": 1, "endpoint_directness": 0.0},
        ),
        ENDPOINT_DIRECT_MOTIF: CausalObservabilityTopologySpec(
            motif_id=ENDPOINT_DIRECT_MOTIF,
            world_topology=GRAPH_ENDPOINT_DIRECT_CAUSE_WORLD_TOPOLOGY,
            sweep_variant_id="endpoint_direct_cause_null",
            expected_response_region=ResponseRegion.NULL_OR_COMPETITIVE_ENDPOINT,
            graph=CausalGraphSpec(
                nodes=(
                    CausalObservabilityVariable("endpoint_action", "action", True),
                    CausalObservabilityVariable(
                        "endpoint_outcome",
                        "outcome",
                        True,
                        ("observable_endpoint_event",),
                    ),
                ),
                edges=(EdgeSpec("endpoint_action", "endpoint_outcome"),),
            ),
            topology_tags=("endpoint_direct", "no_boundary_mediator"),
            hypothesis_targets=(
                "H_BEB_ENDPOINT_SUFFICIENT_NULL_REGION",
                "H_BEB_TOPOLOGY_CONDITIONED_VALIDITY",
            ),
            metric_expectations={
                "root_cause_hit_at_3": "trajectory_competitive",
                "divergence_localization_accuracy": "trajectory_high",
            },
            generator_params={"mediator_depth": 0, "endpoint_directness": 1.0},
        ),
        COLLIDER_CONFOUNDED_MOTIF: CausalObservabilityTopologySpec(
            motif_id=COLLIDER_CONFOUNDED_MOTIF,
            world_topology=GRAPH_COLLIDER_CONFOUNDED_WORLD_TOPOLOGY,
            sweep_variant_id="collider_confounded_failure",
            expected_response_region=ResponseRegion.QUALIFYING_OR_NEGATIVE_BOUNDARY_SCOPE,
            graph=CausalGraphSpec(
                nodes=(
                    CausalObservabilityVariable(
                        "latent_confounder",
                        "confounder",
                        False,
                        ("latent_boundary_event",),
                    ),
                    CausalObservabilityVariable(
                        "boundary_process",
                        "mediator",
                        True,
                        ("observable_boundary_event",),
                    ),
                    CausalObservabilityVariable(
                        "endpoint_outcome",
                        "outcome",
                        True,
                        ("observable_endpoint_event",),
                    ),
                    CausalObservabilityVariable(
                        "observed_collider", "proxy", True, ("proxy_event",)
                    ),
                ),
                edges=(
                    EdgeSpec("latent_confounder", "boundary_process"),
                    EdgeSpec("latent_confounder", "endpoint_outcome"),
                    EdgeSpec("boundary_process", "observed_collider"),
                    EdgeSpec("endpoint_outcome", "observed_collider"),
                ),
            ),
            topology_tags=("collider", "hidden_confounder", "selection_bias"),
            hypothesis_targets=(
                "H_BEB_COLLIDER_CONFOUNDED_FAILURE_REGION",
                "H_BEB_TOPOLOGY_CONDITIONED_VALIDITY",
            ),
            metric_expectations={
                "root_cause_mrr": "boundary_not_universal",
                "divergence_localization_accuracy": "endpoint_can_remain_easy",
            },
            generator_params={"hidden_confounder_count": 1, "collider_present": True},
        ),
        OBSERVABILITY_GAP_MOTIF: CausalObservabilityTopologySpec(
            motif_id=OBSERVABILITY_GAP_MOTIF,
            world_topology=GRAPH_OBSERVABILITY_GAP_WORLD_TOPOLOGY,
            sweep_variant_id="observability_gap",
            expected_response_region=ResponseRegion.OBSERVABILITY_SCOPE_QUALIFIER,
            graph=CausalGraphSpec(
                nodes=(
                    CausalObservabilityVariable("context", "context", True),
                    CausalObservabilityVariable(
                        "boundary_root",
                        "mediator",
                        True,
                        ("observable_boundary_event",),
                    ),
                    CausalObservabilityVariable(
                        "latent_boundary_event",
                        "mediator",
                        False,
                        ("latent_boundary_event",),
                    ),
                    CausalObservabilityVariable(
                        "endpoint_outcome",
                        "outcome",
                        True,
                        ("observable_endpoint_event",),
                    ),
                    CausalObservabilityVariable(
                        "observed_boundary_proxy",
                        "proxy",
                        True,
                        ("proxy_event", "observable_boundary_event"),
                    ),
                ),
                edges=(
                    EdgeSpec("context", "boundary_root"),
                    EdgeSpec("boundary_root", "latent_boundary_event"),
                    EdgeSpec("latent_boundary_event", "endpoint_outcome"),
                    EdgeSpec("boundary_root", "observed_boundary_proxy"),
                ),
            ),
            topology_tags=(
                "boundary_mediator",
                "latent_boundary_event",
                "observed_proxy_bottleneck",
            ),
            hypothesis_targets=(
                "H_BEB_OBSERVABILITY_SCOPE_CONDITION",
                "H_BEB_TOPOLOGY_CONDITIONED_VALIDITY",
            ),
            metric_expectations={
                "root_cause_mrr": "boundary_advantage_requires_observability",
                "negative_control_rejection_rate": "omission_should_be_detected",
            },
            generator_params={"mediator_depth": 2, "boundary_event_observable": False},
        ),
    }


def causal_observability_topology_by_motif(
    motif_id: str,
) -> CausalObservabilityTopologySpec:
    specs = causal_observability_topology_specs()
    try:
        return specs[str(motif_id)]
    except KeyError as exc:
        raise ValueError(
            f"unknown Causal Observability Atlas topology motif: {motif_id!r}"
        ) from exc


def causal_observability_topology_by_world_topology(
    world_topology: str,
) -> CausalObservabilityTopologySpec | None:
    for spec in causal_observability_topology_specs().values():
        if spec.world_topology == str(world_topology):
            return spec
    return None


def derive_causal_observability_region_from_edges(
    hidden_edges: set[tuple[str, str]],
) -> str:
    for spec in causal_observability_topology_specs().values():
        if set(spec.hidden_edges).issubset(hidden_edges):
            return spec.motif_id
    return "unclassified"


def causal_observability_response_alignment(
    response_region: str,
    *,
    metric_lookup: Any,
) -> str:
    def mean(variant: str, policy: str, metric: str) -> float:
        return float(metric_lookup(variant, policy, metric) or 0.0)

    if response_region == "positive_boundary_advantage":
        variant = causal_observability_topology_by_motif(
            BOUNDARY_MEDIATOR_MOTIF
        ).sweep_variant_id
        return (
            "aligned"
            if mean(variant, "boundary_complete_eventization", "root_cause_mrr")
            > mean(variant, "trajectory_level_logging", "root_cause_mrr")
            > mean(variant, "structured_trace_context_logging", "root_cause_mrr")
            > mean(variant, "partial_eventization", "root_cause_mrr")
            else "diverged"
        )
    if response_region == "null_or_competitive_endpoint":
        variant = causal_observability_topology_by_motif(
            ENDPOINT_DIRECT_MOTIF
        ).sweep_variant_id
        return (
            "aligned"
            if mean(variant, "trajectory_level_logging", "root_cause_hit_at_3") >= 0.95
            and mean(
                variant, "trajectory_level_logging", "divergence_localization_accuracy"
            )
            >= 0.95
            and mean(variant, "structured_trace_context_logging", "root_cause_hit_at_1")
            >= 0.95
            else "diverged"
        )
    if response_region == "qualifying_or_negative_boundary_scope":
        variant = causal_observability_topology_by_motif(
            COLLIDER_CONFOUNDED_MOTIF
        ).sweep_variant_id
        return (
            "aligned"
            if mean(variant, "structured_trace_context_logging", "root_cause_mrr")
            >= mean(variant, "boundary_complete_eventization", "root_cause_mrr")
            and mean(
                variant, "trajectory_level_logging", "divergence_localization_accuracy"
            )
            > mean(
                variant,
                "boundary_complete_eventization",
                "divergence_localization_accuracy",
            )
            and mean(variant, "trajectory_level_logging", "root_cause_mrr")
            < mean(variant, "boundary_complete_eventization", "root_cause_mrr")
            else "diverged"
        )
    if response_region == "observability_scope_qualifier":
        variant = causal_observability_topology_by_motif(
            OBSERVABILITY_GAP_MOTIF
        ).sweep_variant_id
        return (
            "aligned"
            if mean(variant, "structured_trace_context_logging", "root_cause_mrr")
            >= mean(variant, "boundary_complete_eventization", "root_cause_mrr")
            and mean(variant, "partial_eventization", "negative_control_rejection_rate")
            >= 0.95
            else "diverged"
        )
    return "not_measured"
