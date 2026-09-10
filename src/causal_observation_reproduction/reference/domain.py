from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class ResponseRegion(StrEnum):
    POSITIVE_BOUNDARY_ADVANTAGE = "positive_boundary_advantage"
    NULL_OR_COMPETITIVE_ENDPOINT = "null_or_competitive_endpoint"
    QUALIFYING_OR_NEGATIVE_BOUNDARY_SCOPE = "qualifying_or_negative_boundary_scope"
    OBSERVABILITY_SCOPE_QUALIFIER = "observability_scope_qualifier"


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    role: str
    observed: bool
    event_trace_roles: tuple[str, ...] = ()

    @property
    def variable_id(self) -> str:
        return self.node_id

    def as_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["event_trace_roles"] = list(self.event_trace_roles)
        return payload


@dataclass(frozen=True)
class EdgeSpec:
    source: str
    target: str
    edge_type: str = "causal"

    def as_pair(self) -> tuple[str, str]:
        return (self.source, self.target)

    def as_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GraphFeatures:
    node_count: int
    edge_count: int
    mediator_depth: int
    has_boundary_mediator: bool
    has_endpoint_direct_path: bool
    has_latent_confounder: bool
    has_observed_collider: bool
    has_latent_boundary_event: bool
    has_observed_proxy: bool
    boundary_event_observable: bool
    latent_node_count: int
    observed_node_count: int
    distractor_count: int

    def as_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CausalGraphSpec:
    nodes: tuple[NodeSpec, ...]
    edges: tuple[EdgeSpec, ...]
    rule_trace: tuple[str, ...] = ()
    features: GraphFeatures | None = None

    @property
    def hidden_edges(self) -> tuple[tuple[str, str], ...]:
        return tuple(edge.as_pair() for edge in self.edges)

    @property
    def observed_variables(self) -> tuple[str, ...]:
        return tuple(node.node_id for node in self.nodes if node.observed)

    @property
    def latent_variables(self) -> tuple[str, ...]:
        return tuple(node.node_id for node in self.nodes if not node.observed)

    @property
    def action_variables(self) -> tuple[str, ...]:
        return tuple(node.node_id for node in self.nodes if node.role == "action")

    @property
    def outcome_variables(self) -> tuple[str, ...]:
        return tuple(node.node_id for node in self.nodes if node.role == "outcome")

    def event_trace_variables(self, trace_role: str) -> tuple[str, ...]:
        return tuple(
            node.node_id
            for node in self.nodes
            if trace_role in set(node.event_trace_roles)
        )

    def as_payload(self) -> dict[str, Any]:
        return {
            "nodes": [node.as_payload() for node in self.nodes],
            "edges": [edge.as_payload() for edge in self.edges],
            "rule_trace": list(self.rule_trace),
            "features": self.features.as_payload() if self.features else {},
        }


@dataclass(frozen=True)
class TopologySpec:
    motif_id: str
    world_topology: str
    sweep_variant_id: str
    expected_response_region: ResponseRegion
    graph: CausalGraphSpec
    topology_tags: tuple[str, ...]
    hypothesis_targets: tuple[str, ...]
    metric_expectations: dict[str, str]
    generator_params: dict[str, Any]

    @property
    def hidden_edges(self) -> tuple[tuple[str, str], ...]:
        return self.graph.hidden_edges

    @property
    def variables(self) -> tuple[NodeSpec, ...]:
        return self.graph.nodes

    @property
    def observed_variables(self) -> tuple[str, ...]:
        return self.graph.observed_variables

    @property
    def latent_variables(self) -> tuple[str, ...]:
        return self.graph.latent_variables

    @property
    def action_variables(self) -> tuple[str, ...]:
        return self.graph.action_variables

    @property
    def outcome_variables(self) -> tuple[str, ...]:
        return self.graph.outcome_variables

    @property
    def expected_evidence_region(self) -> str:
        return str(self.expected_response_region)

    def event_trace_variables(self, trace_role: str) -> tuple[str, ...]:
        return self.graph.event_trace_variables(trace_role)


@dataclass(frozen=True)
class ObservationSurfaceSpec:
    surface_id: str
    observation_modes: frozenset[str]
    distractor_count: int
    observability_tags: tuple[str, ...]
    noise_tags: tuple[str, ...]


@dataclass(frozen=True)
class EventProjectionSpec:
    projection_id: str
    policy_id: str
    source_formalism: str
    target_space: str
    event_types: tuple[str, ...]
    payload_fields: tuple[str, ...]
    observed_variables: tuple[str, ...]
    latent_exclusions: tuple[str, ...]
    dependency_edges_exposed: tuple[tuple[str, str], ...]
    projection_features: dict[str, float]

    def as_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["event_types"] = list(self.event_types)
        payload["payload_fields"] = list(self.payload_fields)
        payload["observed_variables"] = list(self.observed_variables)
        payload["latent_exclusions"] = list(self.latent_exclusions)
        payload["dependency_edges_exposed"] = [
            list(edge) for edge in self.dependency_edges_exposed
        ]
        payload["projection_features"] = dict(self.projection_features)
        return payload
