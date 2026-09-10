from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from causal_observation_reproduction.reference.domain import (
    CausalGraphSpec,
    EdgeSpec,
    GraphFeatures,
    NodeSpec,
)
from causal_observation_reproduction.reference.hashing import (
    stable_json_hash as stable_hash,
)
from causal_observation_reproduction.reference.poscm_dimensions import (
    POSCM_STRUCTURAL_DIMENSION_NAMES,
    PoscmStructuralDimensions,
    summarize_poscm_structural_region,
)
from causal_observation_reproduction.reference.scm_complexity import (
    ScmComplexityTier,
    ScmComplexityVector,
    StructuralEquationConfig,
    build_structural_equation_spec,
    complexity_vector_from_matrix,
    graph_complexity_metrics,
    resolve_complexity_vector,
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

DYNAMIC_TOPOLOGY_REGIONS = (
    "positive_boundary_advantage",
    "null_or_competitive_endpoint",
    "qualifying_or_negative_boundary_scope",
    "observability_scope_qualifier",
)

MOTIF_DESIGN_VECTOR_COLUMNS = POSCM_STRUCTURAL_DIMENSION_NAMES
MotifDesignVector = PoscmStructuralDimensions


@dataclass(frozen=True)
class PoscmTopologyFactoryConfig:
    node_count: int = 9
    edge_density: float = 0.24
    latent_fraction: float = 0.25
    proxy_fraction: float = 0.20
    max_parent_count: int = 3
    distractor_count: int = 5
    seed: int = 17
    horizon: int = 8
    branching_factor: int = 2
    action_count: int = 2
    reward_delay: int = 1
    observation_noise: float = 0.10
    latent_confounding_strength: float = 0.30
    proxy_reliability: float = 0.75
    spurious_correlation_strength: float = 0.20
    complexity_tier: str = ScmComplexityTier.TIER_0_LINEAR.value
    interaction_order: int = 1
    nonlinear_transform_count: int = 0
    temporal_lag: int = 0
    complexity_design_matrix: np.ndarray | None = None
    motif_design_matrix: np.ndarray | None = None

    def as_payload(self) -> dict[str, Any]:
        return {
            "node_count": int(self.node_count),
            "edge_density": float(self.edge_density),
            "latent_fraction": float(self.latent_fraction),
            "proxy_fraction": float(self.proxy_fraction),
            "max_parent_count": int(self.max_parent_count),
            "distractor_count": int(self.distractor_count),
            "seed": int(self.seed),
            "horizon": int(self.horizon),
            "branching_factor": int(self.branching_factor),
            "action_count": int(self.action_count),
            "reward_delay": int(self.reward_delay),
            "observation_noise": float(self.observation_noise),
            "latent_confounding_strength": float(self.latent_confounding_strength),
            "proxy_reliability": float(self.proxy_reliability),
            "spurious_correlation_strength": float(self.spurious_correlation_strength),
            "complexity_tier": str(self.complexity_tier),
            "interaction_order": int(self.interaction_order),
            "nonlinear_transform_count": int(self.nonlinear_transform_count),
            "temporal_lag": int(self.temporal_lag),
            "complexity_design_matrix": _matrix_payload(self.complexity_design_matrix),
            "motif_design_matrix": _matrix_payload(self.motif_design_matrix),
        }


@dataclass(frozen=True)
class GeneratedPoscmTopology:
    topology_id: str
    target_region_hint: str
    derived_region_id: str
    derived_region_basis: str
    graph: CausalGraphSpec
    diagnostics: dict[str, Any]
    semantic_world: dict[str, Any]
    generator_config: PoscmTopologyFactoryConfig

    def as_payload(self) -> dict[str, Any]:
        return {
            "topology_id": self.topology_id,
            "target_region_hint": self.target_region_hint,
            "derived_region_id": self.derived_region_id,
            "derived_region_basis": self.derived_region_basis,
            "graph": self.graph.as_payload(),
            "diagnostics": dict(self.diagnostics),
            "semantic_world": dict(self.semantic_world),
            "generator_config": self.generator_config.as_payload(),
        }


def generate_poscm_topology(
    *,
    config: PoscmTopologyFactoryConfig | None = None,
    target_region_hint: str = "positive_boundary_advantage",
    index: int = 0,
) -> GeneratedPoscmTopology:
    cfg = config or PoscmTopologyFactoryConfig()
    target = str(target_region_hint)
    if target not in set(DYNAMIC_TOPOLOGY_REGIONS):
        raise ValueError(f"unknown dynamic topology target region hint: {target!r}")
    rng = random.Random(
        (int(cfg.seed) * 1009) + (int(index) * 9176) + int(stable_hash(target)[:8], 16)
    )
    motif_vector = motif_design_vector_for_config(cfg=cfg, index=index)
    nodes, edges = _base_region_graph(target)
    nodes, edges = _add_auxiliary_topology(
        nodes, edges, cfg=cfg, rng=rng, target_region=target
    )
    nodes, edges = apply_motif_design_vector(
        nodes,
        edges,
        vector=motif_vector,
        target_region=target,
    )
    graph = CausalGraphSpec(
        nodes=tuple(nodes),
        edges=tuple(edges),
        rule_trace=(
            "dynamic_topology_factory",
            target,
            summarize_motif_region(motif_vector),
        ),
        features=_graph_features(
            nodes, edges, distractor_count=int(round(motif_vector.distractor_count))
        ),
    )
    diagnostics = graph.features.as_payload() if graph.features else {}
    diagnostics["motif_region_label"] = summarize_motif_region(motif_vector)
    diagnostics["motif_design_vector"] = motif_vector.as_payload()
    derived_region_id, derived_region_basis = predict_response_region_from_features(
        diagnostics
    )
    semantic_world = _semantic_world_payload(
        graph=graph,
        diagnostics=diagnostics,
        cfg=cfg,
        motif_vector=motif_vector,
        target_region=target,
        index=index,
    )
    topology_id = (
        "generated-poscm-topology-"
        + stable_hash(
            {
                "target_region_hint": target,
                "index": int(index),
                "config": cfg.as_payload(),
                "nodes": [node.as_payload() for node in nodes],
                "edges": [edge.as_payload() for edge in edges],
                "semantic_world": semantic_world,
            }
        )[:16]
    )
    return GeneratedPoscmTopology(
        topology_id=topology_id,
        target_region_hint=target,
        derived_region_id=derived_region_id,
        derived_region_basis=derived_region_basis,
        graph=graph,
        diagnostics=diagnostics,
        semantic_world=semantic_world,
        generator_config=cfg,
    )


def validate_generated_poscm_topology(payload: dict[str, Any]) -> dict[str, bool]:
    graph = dict(payload.get("graph") or {})
    nodes = [dict(node) for node in graph.get("nodes") or []]
    edges = [dict(edge) for edge in graph.get("edges") or []]
    node_ids = {str(node.get("node_id") or "") for node in nodes}
    diagnostics = dict(payload.get("diagnostics") or {})
    derived_region, _ = predict_response_region_from_features(diagnostics)
    semantic_world = dict(payload.get("semantic_world") or {})
    return {
        "has_topology_id": bool(str(payload.get("topology_id") or "")),
        "has_nodes": bool(nodes),
        "has_edges": bool(edges),
        "node_ids_unique": len(node_ids) == len(nodes),
        "edges_reference_nodes": all(
            str(edge.get("source") or "") in node_ids
            and str(edge.get("target") or "") in node_ids
            for edge in edges
        ),
        "is_acyclic": _is_acyclic(
            node_ids,
            [(str(edge.get("source")), str(edge.get("target"))) for edge in edges],
        ),
        "has_observed_and_latent_nodes": any(
            bool(node.get("observed")) for node in nodes
        )
        and any(not bool(node.get("observed")) for node in nodes),
        "has_endpoint": any(str(node.get("role") or "") == "outcome" for node in nodes),
        "has_diagnostics": bool(diagnostics),
        "derived_region_matches_diagnostics": derived_region
        == str(payload.get("derived_region_id") or ""),
        "has_semantic_world": bool(semantic_world),
        "semantic_world_valid": all(
            validate_semantic_world_payload(semantic_world).values()
        ),
    }


def _base_region_graph(region: str) -> tuple[list[NodeSpec], list[EdgeSpec]]:
    if region == "positive_boundary_advantage":
        return (
            [
                NodeSpec("context", "context", True),
                NodeSpec(
                    "boundary_0", "mediator", True, ("observable_boundary_event",)
                ),
                NodeSpec(
                    "endpoint_outcome", "outcome", True, ("observable_endpoint_event",)
                ),
                NodeSpec("latent_context_noise", "noise", False),
            ],
            [
                EdgeSpec("context", "boundary_0"),
                EdgeSpec("boundary_0", "endpoint_outcome"),
            ],
        )
    if region == "null_or_competitive_endpoint":
        return (
            [
                NodeSpec("endpoint_action", "action", True),
                NodeSpec(
                    "endpoint_outcome", "outcome", True, ("observable_endpoint_event",)
                ),
                NodeSpec("latent_endpoint_noise", "noise", False),
            ],
            [EdgeSpec("endpoint_action", "endpoint_outcome")],
        )
    if region == "qualifying_or_negative_boundary_scope":
        return (
            [
                NodeSpec(
                    "latent_confounder", "confounder", False, ("latent_boundary_event",)
                ),
                NodeSpec(
                    "boundary_process", "mediator", True, ("observable_boundary_event",)
                ),
                NodeSpec(
                    "endpoint_outcome", "outcome", True, ("observable_endpoint_event",)
                ),
                NodeSpec("observed_collider", "collider", True, ("proxy_event",)),
            ],
            [
                EdgeSpec("latent_confounder", "boundary_process"),
                EdgeSpec("latent_confounder", "endpoint_outcome"),
                EdgeSpec("boundary_process", "observed_collider"),
                EdgeSpec("endpoint_outcome", "observed_collider"),
            ],
        )
    return (
        [
            NodeSpec("context", "context", True),
            NodeSpec("boundary_root", "mediator", True, ("observable_boundary_event",)),
            NodeSpec(
                "latent_boundary_event", "mediator", False, ("latent_boundary_event",)
            ),
            NodeSpec(
                "endpoint_outcome", "outcome", True, ("observable_endpoint_event",)
            ),
            NodeSpec(
                "observed_boundary_proxy",
                "proxy",
                True,
                ("proxy_event", "observable_boundary_event"),
            ),
        ],
        [
            EdgeSpec("context", "boundary_root"),
            EdgeSpec("boundary_root", "latent_boundary_event"),
            EdgeSpec("latent_boundary_event", "endpoint_outcome"),
            EdgeSpec("boundary_root", "observed_boundary_proxy"),
        ],
    )


def _add_auxiliary_topology(
    nodes: list[NodeSpec],
    edges: list[EdgeSpec],
    *,
    cfg: PoscmTopologyFactoryConfig,
    rng: random.Random,
    target_region: str,
) -> tuple[list[NodeSpec], list[EdgeSpec]]:
    out_nodes = list(nodes)
    existing = {node.node_id for node in out_nodes}
    target_node_count = max(len(out_nodes), int(cfg.node_count))
    for index in range(target_node_count - len(out_nodes)):
        node_id = f"aux_{index}"
        if node_id in existing:
            continue
        observed = rng.random() >= float(cfg.latent_fraction)
        role = (
            "proxy"
            if rng.random() < float(cfg.proxy_fraction)
            else rng.choice(["context", "mediator", "noise"])
        )
        trace_roles: tuple[str, ...] = (
            ("proxy_event",) if observed and role == "proxy" else ()
        )
        out_nodes.append(NodeSpec(node_id, role, observed, trace_roles))
    order = {node.node_id: index for index, node in enumerate(out_nodes)}
    existing_edges = {(edge.source, edge.target) for edge in edges}
    out_edges = list(edges)
    for source in out_nodes:
        for target in out_nodes:
            if (
                order[source.node_id] >= order[target.node_id]
                or source.node_id == target.node_id
            ):
                continue
            if source.node_id.startswith("aux_") and not target.node_id.startswith(
                "aux_"
            ):
                continue
            if (
                target_region != "qualifying_or_negative_boundary_scope"
                and not source.observed
            ):
                continue
            if target.role == "outcome" and source.role == "noise":
                continue
            parent_count = sum(1 for edge in out_edges if edge.target == target.node_id)
            if parent_count >= int(cfg.max_parent_count):
                continue
            if (
                target_region != "qualifying_or_negative_boundary_scope"
                and target.observed
                and parent_count >= 1
            ):
                continue
            if (source.node_id, target.node_id) in existing_edges:
                continue
            if rng.random() <= float(cfg.edge_density):
                out_edges.append(EdgeSpec(source.node_id, target.node_id))
                existing_edges.add((source.node_id, target.node_id))
    return out_nodes, out_edges


def apply_motif_design_vector(
    nodes: list[NodeSpec],
    edges: list[EdgeSpec],
    *,
    vector: MotifDesignVector,
    target_region: str,
) -> tuple[list[NodeSpec], list[EdgeSpec]]:
    del target_region  # Retained for source-compatible public call signatures.
    out_nodes = list(nodes)
    out_edges = list(edges)
    existing_nodes = {node.node_id for node in out_nodes}
    existing_edges = {(edge.source, edge.target) for edge in out_edges}
    outcome = _first_node_with_role(out_nodes, "outcome") or "endpoint_outcome"
    source = _first_boundary_or_context_node(out_nodes)

    previous = source
    gate_depth = max(
        0,
        int(round(float(vector.gate_depth)))
        - _existing_path_depth(source, outcome, out_edges),
    )
    for index in range(gate_depth):
        gate_id = f"motif_gate_{index}"
        if gate_id not in existing_nodes:
            out_nodes.append(
                NodeSpec(gate_id, "mediator", True, ("observable_boundary_event",))
            )
            existing_nodes.add(gate_id)
        if previous and (previous, gate_id) not in existing_edges:
            out_edges.append(EdgeSpec(previous, gate_id, "motif_gate"))
            existing_edges.add((previous, gate_id))
        previous = gate_id
    if previous and previous != outcome and (previous, outcome) not in existing_edges:
        out_edges.append(EdgeSpec(previous, outcome, "motif_success_chain"))
        existing_edges.add((previous, outcome))

    trap_count = max(0, int(round(float(vector.distractor_count))))
    if _clamp_unit(float(vector.trap_strength)) <= 0.0:
        trap_count = 0
    for index in range(trap_count):
        proxy_id = f"motif_proxy_trap_{index}"
        blocker_id = f"motif_latent_blocker_{index}"
        collider_id = f"motif_collider_trap_{index}"
        proxy_observed = float(vector.proxy_observability) >= 0.5
        for node in (
            NodeSpec(proxy_id, "proxy", proxy_observed, ("proxy_event",)),
            NodeSpec(blocker_id, "confounder", False, ("latent_boundary_event",)),
            NodeSpec(collider_id, "collider", True, ("proxy_event",)),
        ):
            if node.node_id not in existing_nodes:
                out_nodes.append(node)
                existing_nodes.add(node.node_id)
        for edge in (
            EdgeSpec(source, proxy_id, "motif_spurious_proxy"),
            EdgeSpec(proxy_id, collider_id, "motif_spurious_proxy"),
            EdgeSpec(blocker_id, collider_id, "motif_collider_block"),
            EdgeSpec(collider_id, outcome, "motif_delayed_penalty"),
        ):
            if (edge.source, edge.target) not in existing_edges:
                out_edges.append(edge)
                existing_edges.add((edge.source, edge.target))
    return out_nodes, out_edges


def _graph_features(
    nodes: list[NodeSpec], edges: list[EdgeSpec], *, distractor_count: int
) -> GraphFeatures:
    node_by_id = {node.node_id: node for node in nodes}
    edge_pairs = [edge.as_pair() for edge in edges]
    children: dict[str, list[str]] = {}
    parents: dict[str, list[str]] = {}
    for source, target in edge_pairs:
        children.setdefault(source, []).append(target)
        parents.setdefault(target, []).append(source)
    latent_nodes = {node.node_id for node in nodes if not node.observed}
    observed_nodes = {node.node_id for node in nodes if node.observed}
    endpoint_nodes = {node.node_id for node in nodes if node.role == "outcome"}
    boundary_nodes = {
        node.node_id
        for node in nodes
        if "observable_boundary_event" in set(node.event_trace_roles)
    }
    proxy_nodes = {
        node.node_id
        for node in nodes
        if node.role == "proxy" or "proxy_event" in set(node.event_trace_roles)
    }
    latent_boundary_nodes = {
        node.node_id
        for node in nodes
        if "latent_boundary_event" in set(node.event_trace_roles)
    }
    action_nodes = {node.node_id for node in nodes if node.role == "action"}
    collider_nodes = {
        node.node_id
        for node in nodes
        if node.role == "collider"
        or (node.observed and len(parents.get(node.node_id, [])) >= 2)
    }
    latent_confounders = {
        node_id
        for node_id in latent_nodes
        if node_by_id[node_id].role == "confounder"
        or len(children.get(node_id, [])) >= 2
    }
    has_boundary_path = any(
        _has_path(boundary_node, endpoint, children)
        for boundary_node in boundary_nodes
        for endpoint in endpoint_nodes
    )
    has_endpoint_direct = any(
        source in action_nodes and target in endpoint_nodes
        for source, target in edge_pairs
    )
    return GraphFeatures(
        node_count=len(nodes),
        edge_count=len(edge_pairs),
        mediator_depth=_mediator_depth(boundary_nodes, endpoint_nodes, children),
        has_boundary_mediator=has_boundary_path,
        has_endpoint_direct_path=has_endpoint_direct,
        has_latent_confounder=bool(latent_confounders),
        has_observed_collider=bool(collider_nodes),
        has_latent_boundary_event=bool(latent_boundary_nodes),
        has_observed_proxy=bool(proxy_nodes & observed_nodes),
        boundary_event_observable=bool(boundary_nodes & observed_nodes),
        latent_node_count=len(latent_nodes),
        observed_node_count=len(observed_nodes),
        distractor_count=int(distractor_count),
    )


def _semantic_world_payload(
    *,
    graph: CausalGraphSpec,
    diagnostics: dict[str, Any],
    cfg: PoscmTopologyFactoryConfig,
    motif_vector: MotifDesignVector,
    target_region: str,
    index: int,
) -> dict[str, Any]:
    nodes = list(graph.nodes)
    edges = list(graph.edges)
    node_ids = [node.node_id for node in nodes]
    action_variables = [f"action_{idx}" for idx in range(max(1, int(cfg.action_count)))]
    outcome_variables = list(graph.outcome_variables) or ["endpoint_outcome"]
    parent_map: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    for edge in edges:
        parent_map.setdefault(edge.target, []).append(edge.source)
    transition_variables = [
        node.node_id
        for node in nodes
        if node.role not in {"action"} and node.node_id in node_ids
    ]
    transition_model: list[TransitionModelRowSpec] = []
    complexity_vector = _complexity_vector_for_config(cfg=cfg, index=index)
    equation_config = StructuralEquationConfig(
        complexity_tier=str(cfg.complexity_tier),
        interaction_order=int(cfg.interaction_order),
        nonlinear_transform_count=int(cfg.nonlinear_transform_count),
        temporal_lag=int(cfg.temporal_lag),
        threshold_count=1
        if str(cfg.complexity_tier)
        in {
            ScmComplexityTier.TIER_2_SMOOTH_NONLINEAR.value,
            ScmComplexityTier.TIER_3_LATENT_CONFOUNDING.value,
            ScmComplexityTier.TIER_4_TEMPORAL.value,
            ScmComplexityTier.TIER_5_COMPOSITIONAL.value,
        }
        else 0,
        complexity_vector=complexity_vector,
    )
    for node in nodes:
        if node.node_id not in transition_variables:
            continue
        action_inputs = _action_inputs_for_node(
            node=node,
            action_variables=action_variables,
            outcome_variables=outcome_variables,
        )
        latent_noise_inputs = _latent_noise_inputs_for_node(
            node=node, nodes=nodes, parent_map=parent_map
        )
        equation_spec = build_structural_equation_spec(
            variable_id=node.node_id,
            parent_variables=tuple(sorted(parent_map.get(node.node_id, []))),
            action_inputs=tuple(action_inputs),
            latent_noise_inputs=tuple(latent_noise_inputs),
            config=equation_config,
        )
        transition_model.append(
            TransitionModelRowSpec(
                variable_id=node.node_id,
                parents=tuple(sorted(parent_map.get(node.node_id, []))),
                action_inputs=tuple(action_inputs),
                latent_noise_inputs=tuple(latent_noise_inputs),
                equation=_transition_equation(node),
                structural_equation=equation_spec,
                transition_delay=int(cfg.reward_delay)
                if node.node_id in set(outcome_variables)
                else 0,
            )
        )
    observation_model = _observation_model_payload(
        graph=graph,
        cfg=cfg,
        complexity_vector=complexity_vector,
        motif_vector=motif_vector,
    )
    reward_model = RewardModelSpec(
        reward_id="semantic_reward",
        outcome_variables=tuple(outcome_variables),
        action_variables=tuple(action_variables),
        reward_delay=max(0, int(round(float(motif_vector.reward_delay)))),
        success_reward=1.0,
        failure_reward=-0.05 if float(cfg.observation_noise) > 0.0 else 0.0,
        success_condition="any outcome variable reaches its positive terminal state",
        credit_assignment_target=_credit_assignment_target(target_region),
    )
    difficulty_vector = _difficulty_vector_payload(
        graph=graph,
        diagnostics=diagnostics,
        cfg=cfg,
        complexity_vector=complexity_vector,
    )
    complexity = _complexity_payload(
        graph=graph,
        transition_model=tuple(transition_model),
        cfg=cfg,
        complexity_vector=complexity_vector,
    )
    semantic_world = SemanticWorldSpec(
        state_variables=tuple(
            StateVariableSpec(
                variable_id=node.node_id, role=node.role, observed=bool(node.observed)
            )
            for node in nodes
        ),
        action_space=ActionSpaceSpec(
            action_variables=tuple(action_variables),
            branching_factor=max(1, int(cfg.branching_factor)),
        ),
        transition_model=tuple(transition_model),
        reward_model=reward_model,
        observation_model=observation_model,
        interventions=tuple(_intervention_payloads(graph=graph, cfg=cfg)),
        difficulty_vector=difficulty_vector,
        complexity={**complexity, "motif_design_vector": motif_vector.as_payload()},
        target_region=target_region,
        index=int(index),
    )
    return semantic_world.as_payload()


def _action_inputs_for_node(
    *,
    node: NodeSpec,
    action_variables: list[str],
    outcome_variables: list[str],
) -> list[str]:
    if node.role in {"mediator", "outcome"} or node.node_id in set(outcome_variables):
        return list(action_variables)
    return []


def _latent_noise_inputs_for_node(
    *,
    node: NodeSpec,
    nodes: list[NodeSpec],
    parent_map: dict[str, list[str]],
) -> list[str]:
    latent_nodes = {item.node_id for item in nodes if not item.observed}
    parents = set(parent_map.get(node.node_id, []))
    return sorted(parents & latent_nodes)


def _transition_equation(node: NodeSpec) -> str:
    if node.role == "outcome":
        return "thresholded_parent_action_noisy_or"
    if node.role == "proxy":
        return "noisy_parent_proxy_sample"
    if node.role == "collider":
        return "collider_parent_conjunction"
    if node.role == "noise":
        return "bernoulli_latent_disturbance"
    return "parent_action_noisy_or"


def _observation_model_payload(
    *,
    graph: CausalGraphSpec,
    cfg: PoscmTopologyFactoryConfig,
    complexity_vector: ScmComplexityVector,
    motif_vector: MotifDesignVector,
) -> ObservationModelSpec:
    vector = complexity_vector.as_payload()
    event_emitters = {
        trace_role: list(graph.event_trace_variables(trace_role))
        for trace_role in (
            "observable_boundary_event",
            "latent_boundary_event",
            "proxy_event",
            "observable_endpoint_event",
        )
    }
    return ObservationModelSpec(
        observed_variables=tuple(graph.observed_variables),
        latent_variables=tuple(graph.latent_variables),
        observation_noise=_clamp_unit(
            max(
                float(cfg.observation_noise),
                vector["noise_scale"],
                float(motif_vector.noise_scale),
            )
        ),
        proxy_reliability=_clamp_unit(
            min(
                float(cfg.proxy_reliability),
                float(motif_vector.proxy_observability),
                1.0 - (0.5 * vector["partial_observability"]),
            )
        ),
        spurious_correlation_strength=_clamp_unit(
            max(
                float(cfg.spurious_correlation_strength),
                float(motif_vector.proxy_strength),
            )
        ),
        event_emitters=event_emitters,
    )


def _intervention_payloads(
    *, graph: CausalGraphSpec, cfg: PoscmTopologyFactoryConfig
) -> list[InterventionSpec]:
    boundary_targets = list(graph.event_trace_variables("observable_boundary_event"))
    proxy_targets = list(graph.event_trace_variables("proxy_event"))
    latent_targets = list(graph.latent_variables)
    edge_targets = [
        list(edge.as_pair()) for edge in graph.edges[: max(1, min(3, len(graph.edges)))]
    ]
    interventions = [
        InterventionSpec(
            intervention_id="mask_boundary_events",
            intervention_type="mask_event_role",
            targets=tuple(boundary_targets),
            parameters={
                "mask_probability": min(1.0, 0.5 + float(cfg.observation_noise))
            },
        ),
        InterventionSpec(
            intervention_id="remove_proxy_channel",
            intervention_type="drop_observation_channel",
            targets=tuple(proxy_targets),
            parameters={"proxy_reliability": 0.0},
        ),
        InterventionSpec(
            intervention_id="reveal_latent_state",
            intervention_type="add_observation_channel",
            targets=tuple(latent_targets),
            parameters={"observation_noise": 0.0},
        ),
        InterventionSpec(
            intervention_id="perturb_edge_strength",
            intervention_type="edge_strength_shift",
            targets=tuple(tuple(edge) for edge in edge_targets),
            parameters={"delta": round(float(cfg.latent_confounding_strength), 6)},
        ),
    ]
    return [item for item in interventions if item.targets]


def _difficulty_vector_payload(
    *,
    graph: CausalGraphSpec,
    diagnostics: dict[str, Any],
    cfg: PoscmTopologyFactoryConfig,
    complexity_vector: ScmComplexityVector,
) -> dict[str, Any]:
    node_count = max(
        1.0, float(diagnostics.get("node_count") or len(graph.nodes) or 1.0)
    )
    edge_count = float(diagnostics.get("edge_count") or len(graph.edges))
    max_edges = max(1.0, node_count * (node_count - 1.0) / 2.0)
    vector = complexity_vector.as_payload()
    return {
        "horizon": float(max(1, int(cfg.horizon))),
        "branching_factor": float(max(1, int(cfg.branching_factor))),
        "action_count": float(max(1, int(cfg.action_count))),
        "reward_delay": float(max(0, int(cfg.reward_delay))),
        "observation_noise": _clamp_unit(float(cfg.observation_noise)),
        "latent_confounding_strength": _clamp_unit(
            float(cfg.latent_confounding_strength)
        ),
        "proxy_reliability": _clamp_unit(float(cfg.proxy_reliability)),
        "spurious_correlation_strength": _clamp_unit(
            float(cfg.spurious_correlation_strength)
        ),
        "topology_edge_density": round(edge_count / max_edges, 6),
        "latent_fraction": round(
            float(diagnostics.get("latent_node_count") or 0.0) / node_count, 6
        ),
        "complexity_tier": str(cfg.complexity_tier),
        "interaction_order": float(max(1, int(cfg.interaction_order))),
        "nonlinear_transform_count": float(max(0, int(cfg.nonlinear_transform_count))),
        "temporal_lag": float(max(0, int(cfg.temporal_lag))),
        "complexity_vector": vector,
        "motif_design_vector": dict(diagnostics.get("motif_design_vector") or {}),
        "motif_region_label": str(diagnostics.get("motif_region_label") or ""),
        **{f"complexity_{key}": float(value) for key, value in vector.items()},
    }


def _complexity_payload(
    *,
    graph: CausalGraphSpec,
    transition_model: tuple[TransitionModelRowSpec, ...],
    cfg: PoscmTopologyFactoryConfig,
    complexity_vector: ScmComplexityVector,
) -> dict[str, Any]:
    equation_metrics = [
        dict(row.structural_equation.get("metrics") or {}) for row in transition_model
    ]
    graph_metrics = graph_complexity_metrics(
        graph, complexity_tier=str(cfg.complexity_tier)
    )
    return {
        "complexity_tier": str(cfg.complexity_tier),
        "complexity_vector": complexity_vector.as_payload(),
        "graph_metrics": graph_metrics,
        "equation_summary": {
            "equation_count": len(equation_metrics),
            "max_polynomial_degree": int(
                max(
                    (
                        int(item.get("polynomial_degree") or 0)
                        for item in equation_metrics
                    ),
                    default=0,
                )
            ),
            "interaction_term_count": int(
                sum(
                    int(item.get("interaction_term_count") or 0)
                    for item in equation_metrics
                )
            ),
            "nonlinear_transform_count": int(
                sum(
                    int(item.get("nonlinear_transform_count") or 0)
                    for item in equation_metrics
                )
            ),
            "latent_input_count": int(
                sum(
                    int(item.get("latent_input_count") or 0)
                    for item in equation_metrics
                )
            ),
            "max_temporal_lag_depth": int(
                max(
                    (
                        int(item.get("temporal_lag_depth") or 0)
                        for item in equation_metrics
                    ),
                    default=0,
                )
            ),
        },
    }


def _credit_assignment_target(target_region: str) -> str:
    if target_region == "qualifying_or_negative_boundary_scope":
        return "distinguish_collider_proxy_from_causal_boundary"
    if target_region == "observability_scope_qualifier":
        return "recover_latent_boundary_effect_from_proxy"
    if target_region == "null_or_competitive_endpoint":
        return "separate_endpoint_action_from_boundary_context"
    return "attribute_reward_to_boundary_mediator"


def _complexity_vector_for_config(
    *, cfg: PoscmTopologyFactoryConfig, index: int
) -> ScmComplexityVector:
    if cfg.complexity_design_matrix is not None:
        return complexity_vector_from_matrix(
            cfg.complexity_design_matrix, row_index=index
        )
    config = StructuralEquationConfig(
        complexity_tier=str(cfg.complexity_tier),
        interaction_order=int(cfg.interaction_order),
        nonlinear_transform_count=int(cfg.nonlinear_transform_count),
        temporal_lag=int(cfg.temporal_lag),
    )
    return resolve_complexity_vector(config)


def build_motif_design_vectors(matrix: np.ndarray) -> tuple[MotifDesignVector, ...]:
    array = np.asarray(matrix, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError("motif design matrix must be one- or two-dimensional")
    return tuple(_motif_design_vector_from_row(row) for row in array)


def motif_design_vector_for_config(
    *, cfg: PoscmTopologyFactoryConfig, index: int
) -> MotifDesignVector:
    if cfg.motif_design_matrix is not None:
        vectors = build_motif_design_vectors(cfg.motif_design_matrix)
        if not vectors:
            return _default_motif_design_vector(cfg)
        return vectors[int(index) % len(vectors)]
    return _default_motif_design_vector(cfg)


def summarize_motif_region(vector: MotifDesignVector) -> str:
    return summarize_poscm_structural_region(vector)


def _motif_design_vector_from_row(row: np.ndarray) -> MotifDesignVector:
    return MotifDesignVector.from_motif_design_matrix_row(row)


def _default_motif_design_vector(cfg: PoscmTopologyFactoryConfig) -> MotifDesignVector:
    return MotifDesignVector(
        solution_density_target=0.15,
        gate_depth=0.0,
        action_branching_factor=float(max(1, int(cfg.action_count))),
        correct_action_fraction=1.0 / float(max(1, int(cfg.action_count))),
        distractor_count=0.0,
        distractor_depth=1.0,
        trap_strength=0.0,
        proxy_strength=float(cfg.spurious_correlation_strength),
        proxy_observability=float(cfg.proxy_reliability),
        collider_strength=0.0,
        delayed_penalty_strength=0.0,
        latent_confounding_strength=float(cfg.latent_confounding_strength),
        noise_scale=float(cfg.observation_noise),
        reward_delay=float(max(0, int(cfg.reward_delay))),
        observation_mask_fraction=1.0 - float(cfg.latent_fraction),
    )


def _matrix_payload(matrix: np.ndarray | None) -> list[list[float]] | None:
    if matrix is None:
        return None
    array = np.asarray(matrix, dtype=float)
    if array.ndim == 1:
        return [[float(value) for value in array]]
    if array.ndim == 2:
        return [[float(value) for value in row] for row in array]
    raise ValueError("SCM complexity design matrix must be one- or two-dimensional")


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _has_path(source: str, target: str, children: dict[str, list[str]]) -> bool:
    frontier = [source]
    seen: set[str] = set()
    while frontier:
        node = frontier.pop(0)
        if node == target:
            return True
        if node in seen:
            continue
        seen.add(node)
        frontier.extend(children.get(node, []))
    return False


def _first_node_with_role(nodes: list[NodeSpec], role: str) -> str:
    return str(next((node.node_id for node in nodes if node.role == role), ""))


def _first_boundary_or_context_node(nodes: list[NodeSpec]) -> str:
    return str(
        next(
            (
                node.node_id
                for node in nodes
                if "observable_boundary_event" in set(node.event_trace_roles)
                or node.role in {"context", "action", "mediator"}
            ),
            nodes[0].node_id if nodes else "",
        )
    )


def _existing_path_depth(source: str, target: str, edges: list[EdgeSpec]) -> int:
    if not source or not target:
        return 0
    children: dict[str, list[str]] = {}
    for edge in edges:
        children.setdefault(edge.source, []).append(edge.target)
    frontier: list[tuple[str, int]] = [(source, 0)]
    seen: set[str] = set()
    while frontier:
        node, depth = frontier.pop(0)
        if node in seen:
            continue
        seen.add(node)
        if node == target:
            return int(depth)
        frontier.extend((child, depth + 1) for child in children.get(node, []))
    return 0


def _mediator_depth(
    boundary_nodes: set[str], endpoint_nodes: set[str], children: dict[str, list[str]]
) -> int:
    max_depth = 0
    for boundary_node in boundary_nodes:
        frontier = [(boundary_node, 0)]
        seen: set[str] = set()
        while frontier:
            node, depth = frontier.pop(0)
            if node in seen:
                continue
            seen.add(node)
            if node in endpoint_nodes:
                max_depth = max(max_depth, depth)
                continue
            for child in children.get(node, []):
                frontier.append((child, depth + 1))
    return max_depth


def _is_acyclic(node_ids: set[str], edges: list[tuple[str, str]]) -> bool:
    children: dict[str, list[str]] = {}
    for source, target in edges:
        children.setdefault(source, []).append(target)
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return False
        if node in visited:
            return True
        visiting.add(node)
        for child in children.get(node, []):
            if not visit(child):
                return False
        visiting.remove(node)
        visited.add(node)
        return True

    return all(visit(node) for node in node_ids)


def predict_response_region_from_features(features: dict[str, Any]) -> tuple[str, str]:
    if bool(features.get("has_latent_confounder")) and bool(
        features.get("has_observed_collider")
    ):
        return (
            "qualifying_or_negative_boundary_scope",
            "latent_confounder_observed_collider",
        )
    if bool(features.get("has_latent_boundary_event")) and bool(
        features.get("has_observed_proxy")
    ):
        return "observability_scope_qualifier", "latent_boundary_observed_proxy"
    if bool(features.get("has_endpoint_direct_path")) and not bool(
        features.get("has_boundary_mediator")
    ):
        return "null_or_competitive_endpoint", "endpoint_direct_without_boundary"
    if bool(features.get("has_boundary_mediator")) and bool(
        features.get("boundary_event_observable")
    ):
        return "positive_boundary_advantage", "observable_boundary_mediator"
    if bool(features.get("has_endpoint_direct_path")):
        return "null_or_competitive_endpoint", "endpoint_direct_fallback"
    return "observability_scope_qualifier", "observability_gap_fallback"
