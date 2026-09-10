"""Controlled POSCM families for the causal-quotient novelty gate.

The random topology factory supplies structural variety, while this module
adds a small, explicit decision-relevant latent challenge.  Its purpose is to
construct *known* positive, null, and invalid quotient regimes before an
experiment is run.  Oracle action signatures remain in a separate sidecar;
the semantic world and the deployment manifest never contain their labels.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

from causal_observation_reproduction.reference.hashing import (
    stable_json_hash as stable_hash,
)
from causal_observation_reproduction.reference.interventional_signature import (
    build_generated_poscm_reference_quotient_sidecar,
    enumerate_binary_contexts,
)
from causal_observation_reproduction.reference.topology_factory import (
    PoscmTopologyFactoryConfig,
    generate_poscm_topology,
)

QuotientRegime = Literal["positive", "null", "invalid"]


@dataclass(frozen=True)
class CausalQuotientFamilySpec:
    """Parameters for one finite, decision-relative quotient family.

    ``probe_cost``, ``probe_reliability``, ``sensing_budget``, and
    ``alias_persistence`` are protocol parameters: they are preserved in the
    deployment manifest but are not yet simulated as observation actions.
    This makes the boundary between a validated quotient label and a future
    sequential sensing environment explicit.
    """

    regime: QuotientRegime
    horizon: int = 4
    reward_delay: int = 1
    continuation_dependency_depth: int = 1
    action_count: int = 2
    distractor_count: int = 0
    observation_noise: float = 0.0
    probe_cost: float = 0.10
    probe_reliability: float = 1.0
    sensing_budget: int = 1
    alias_persistence: int = 1
    topology_family: str = "positive_boundary_advantage"
    topology_instance_id: str = ""
    seed: int = 17

    def __post_init__(self) -> None:
        if self.regime not in {"positive", "null", "invalid"}:
            raise ValueError("regime must be positive, null, or invalid")
        if self.action_count < 2:
            raise ValueError("action_count must be at least two")
        if (
            self.horizon < 1
            or self.reward_delay < 0
            or self.reward_delay >= self.horizon
        ):
            raise ValueError(
                "reward_delay must be nonnegative and smaller than horizon"
            )
        if not 1 <= self.continuation_dependency_depth <= self.horizon:
            raise ValueError("continuation_dependency_depth must be within the horizon")
        if (
            self.distractor_count < 0
            or self.sensing_budget < 0
            or self.alias_persistence < 1
        ):
            raise ValueError(
                "counts and persistence must be nonnegative (persistence at least one)"
            )
        if not 0.0 <= self.observation_noise <= 1.0:
            raise ValueError("observation_noise must lie in [0, 1]")
        if self.probe_cost < 0.0 or not 0.0 <= self.probe_reliability <= 1.0:
            raise ValueError(
                "probe cost must be nonnegative and reliability must lie in [0, 1]"
            )

    def deployment_payload(self) -> dict[str, Any]:
        """Return protocol settings that may be visible to an environment."""
        return {
            "regime": self.regime,
            "horizon": self.horizon,
            "reward_delay": self.reward_delay,
            "continuation_dependency_depth": self.continuation_dependency_depth,
            "action_count": self.action_count,
            "distractor_count": self.distractor_count,
            "observation_noise": self.observation_noise,
            "probe_cost": self.probe_cost,
            "probe_reliability": self.probe_reliability,
            "sensing_budget": self.sensing_budget,
            "alias_persistence": self.alias_persistence,
            "topology_family": self.topology_family,
            "topology_instance_id": self.topology_instance_id,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class GeneratedCausalQuotientFamily:
    """One generated semantic world plus its non-deployment oracle sidecar."""

    family_id: str
    semantic_world: dict[str, Any]
    deployment_manifest: dict[str, Any]
    scorer_sidecar: dict[str, Any]

    def as_payload(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "semantic_world": deepcopy(self.semantic_world),
            "deployment_manifest": deepcopy(self.deployment_manifest),
            "scorer_sidecar": deepcopy(self.scorer_sidecar),
        }


def generate_causal_quotient_family(
    *, spec: CausalQuotientFamilySpec, index: int = 0
) -> GeneratedCausalQuotientFamily:
    """Generate and oracle-check a controlled quotient family.

    Positive and invalid regimes contain two latent contexts with different
    action requirements.  The positive observation map keeps them separate;
    the invalid map deliberately aliases them.  A null regime gives both
    contexts the same continuation requirement.  All three claims are then
    checked by the finite-horizon action-signature oracle rather than trusted
    merely because they were requested.
    """
    topology = generate_poscm_topology(
        config=PoscmTopologyFactoryConfig(
            node_count=max(6, 6 + int(spec.distractor_count)),
            distractor_count=int(spec.distractor_count),
            action_count=int(spec.action_count),
            horizon=int(spec.horizon),
            reward_delay=int(spec.reward_delay),
            observation_noise=float(spec.observation_noise),
            seed=int(spec.seed),
        ),
        target_region_hint=str(spec.topology_family),
        index=int(index),
    )
    semantic_world = deepcopy(topology.semantic_world)
    context_variable = _first_latent_variable(semantic_world)
    _install_contextual_action_target(
        semantic_world, spec=spec, context_variable=context_variable
    )
    contexts = enumerate_binary_contexts(
        semantic_world, variable_ids=[context_variable], max_contexts=2
    )
    proposed_classes = _proposed_observation_classes(spec.regime, contexts)
    world_id = (
        "quotient-family-world-"
        + stable_hash(
            {
                "world": semantic_world,
                "spec": spec.deployment_payload(),
                "index": int(index),
            }
        )[:16]
    )
    bridge = {
        "bridge_id": "quotient-family-bridge-" + world_id[-16:],
        "worlds": [{"world_id": world_id, "semantic_world": semantic_world}],
    }
    deployment_manifest = {
        "schema_version": "causal_observation_reproduction.causal_quotient_family.v1",
        "artifact_scope": "deployment_protocol",
        "selection_eligible": True,
        "deployment_visible": True,
        "world_id": world_id,
        "declared_regime": spec.regime,
        # This is the declared *decision query*, not an oracle label or a
        # semantic-world payload.  A controller may use this compact public
        # contract to decide whether a refinement could distinguish different
        # continuation actions, but it never receives latent assignments,
        # reward-model rows, or the scorer sidecar.
        "causal_query": {
            "schema_version": "causal_observation_reproduction.decision_relative_query.v1",
            "target": "context_conditioned_continuation_action",
            "action_value_varies_across_declared_contexts": spec.regime != "null",
        },
        "proposed_observation_classes": proposed_classes,
        "protocol": spec.deployment_payload(),
    }
    scorer_sidecar = build_generated_poscm_reference_quotient_sidecar(
        bridge,
        contexts_by_world={world_id: contexts},
        manifests_by_world={world_id: deployment_manifest},
        horizon=spec.horizon,
        scorer_seed=spec.seed,
    )
    family_id = (
        "causal-quotient-family-"
        + stable_hash({"world_id": world_id, "protocol": spec.deployment_payload()})[
            :16
        ]
    )
    return GeneratedCausalQuotientFamily(
        family_id=family_id,
        semantic_world=semantic_world,
        deployment_manifest=deployment_manifest,
        scorer_sidecar=scorer_sidecar,
    )


def _first_latent_variable(semantic_world: dict[str, Any]) -> str:
    latent = [
        str(value)
        for value in dict(semantic_world.get("observation_model") or {}).get(
            "latent_variables"
        )
        or []
    ]
    if not latent:
        raise ValueError(
            "generated topology needs at least one latent variable for a quotient family"
        )
    return latent[0]


def _install_contextual_action_target(
    semantic_world: dict[str, Any],
    *,
    spec: CausalQuotientFamilySpec,
    context_variable: str,
) -> None:
    """Install a transparent context-conditioned continuation target.

    The evaluator interprets this reward-model extension directly.  It is
    intentionally simple: a family is a controlled benchmark fixture, not a
    claim that arbitrary generated SCMs have this property.
    """
    depth = int(spec.continuation_dependency_depth)
    shared = [0] * depth
    context_one = shared if spec.regime == "null" else [1] * depth
    reward_model = dict(semantic_world.get("reward_model") or {})
    reward_model["context_required_action_sequences"] = [
        {"context": {context_variable: 0.0}, "required_action_sequence": shared},
        {"context": {context_variable: 1.0}, "required_action_sequence": context_one},
    ]
    reward_model["contextual_target_scope"] = "controlled_causal_quotient_family"
    semantic_world["reward_model"] = reward_model


def _proposed_observation_classes(
    regime: QuotientRegime, contexts: dict[str, dict[str, float]]
) -> dict[str, str]:
    context_ids = tuple(sorted(contexts))
    if regime == "positive":
        return {
            context_id: f"refined-{position}"
            for position, context_id in enumerate(context_ids)
        }
    if regime in {"null", "invalid"}:
        return dict.fromkeys(context_ids, "coarse")
    raise AssertionError(f"unreachable quotient regime: {regime!r}")


__all__ = [
    "CausalQuotientFamilySpec",
    "GeneratedCausalQuotientFamily",
    "QuotientRegime",
    "generate_causal_quotient_family",
]
