"""Scorer-only finite-horizon action signatures for generated POSCM worlds.

This module turns the Lean novelty-gate condition into a benchmark-construction
primitive.  A *reference quotient* groups declared latent contexts only when
their finite-horizon action-value vectors agree under paired, seeded rollouts.
It is an oracle label for scoring and split construction, never a field for a
deployed observation, policy, or model-selection feature set.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from math import isfinite
from typing import Any

from causal_observation_reproduction.reference.hashing import (
    stable_json_hash as stable_hash,
)

INTERVENTIONAL_ACTION_SIGNATURE_SCHEMA_VERSION = (
    "causal_observation_reproduction.interventional_action_signature.v1"
)
_DECLARED_REGIMES = frozenset({"positive", "null", "invalid"})


@dataclass(frozen=True)
class InterventionalActionSignature:
    """An oracle action-value vector for one declared latent context."""

    context_id: str
    horizon: int
    action_values: tuple[float, ...]

    @property
    def quotient_label(self) -> str:
        return (
            "signature-"
            + stable_hash(
                {
                    "horizon": int(self.horizon),
                    "action_values": list(self.action_values),
                }
            )[:16]
        )

    def scorer_payload(self) -> dict[str, Any]:
        return {
            "context_id": str(self.context_id),
            "horizon": int(self.horizon),
            "action_values": [float(value) for value in self.action_values],
            "reference_quotient_label": self.quotient_label,
        }


@dataclass(frozen=True)
class ReferenceQuotient:
    """A scorer-only partition induced by finite-horizon action signatures."""

    signatures: tuple[InterventionalActionSignature, ...]
    scorer_seed: int

    @property
    def labels_by_context(self) -> dict[str, str]:
        return {
            signature.context_id: signature.quotient_label
            for signature in self.signatures
        }

    def scorer_payload(self) -> dict[str, Any]:
        return {
            "schema_version": INTERVENTIONAL_ACTION_SIGNATURE_SCHEMA_VERSION,
            "artifact_scope": "scorer_only_oracle",
            "selection_eligible": False,
            "deployment_visible": False,
            "scorer_seed": int(self.scorer_seed),
            "signatures": [signature.scorer_payload() for signature in self.signatures],
            "labels_by_context": self.labels_by_context,
        }


@dataclass(frozen=True)
class DeclaredQuotientRegime:
    declared_regime: str
    resolved_regime: str
    proposed_classes: dict[str, str]
    reference_quotient: ReferenceQuotient

    def scorer_payload(self) -> dict[str, Any]:
        return {
            "schema_version": INTERVENTIONAL_ACTION_SIGNATURE_SCHEMA_VERSION,
            "artifact_scope": "scorer_only_oracle",
            "selection_eligible": False,
            "deployment_visible": False,
            "declared_regime": self.declared_regime,
            "resolved_regime": self.resolved_regime,
            "proposed_observation_classes": dict(self.proposed_classes),
            "reference_quotient": self.reference_quotient.scorer_payload(),
        }


def enumerate_binary_contexts(
    semantic_world: Mapping[str, Any],
    *,
    variable_ids: Sequence[str] | None = None,
    max_contexts: int = 32,
) -> dict[str, dict[str, float]]:
    """Enumerate a bounded declared binary context set for scorer-only use.

    Callers normally select latent variables.  The bound fails closed instead
    of silently sampling or exposing an unbounded latent state space.
    """
    all_variables = [
        str(row.get("variable_id") or "")
        for row in semantic_world.get("state_variables") or []
    ]
    selected = tuple(str(value) for value in (variable_ids or all_variables))
    if not selected or any(value not in set(all_variables) for value in selected):
        raise ValueError(
            "signature contexts must name declared semantic-world state variables"
        )
    count = 2 ** len(selected)
    if count > max(1, int(max_contexts)):
        raise ValueError(
            "declared signature context space exceeds the scorer-only bound"
        )
    return {
        "context-" + "".join(str(bit) for bit in bits): {
            variable_id: float(bit)
            for variable_id, bit in zip(selected, bits, strict=True)
        }
        for bits in product((0, 1), repeat=len(selected))
    }


def derive_finite_horizon_action_signatures(
    semantic_world: Mapping[str, Any],
    *,
    contexts: Mapping[str, Mapping[str, float]],
    horizon: int | None = None,
    scorer_seed: int = 0,
    max_action_sequences: int = 4096,
) -> ReferenceQuotient:
    """Evaluate every first action against a bounded continuation enumeration.

    Each candidate action sequence begins from the same declared context and
    seed.  That paired construction makes the label an interventional oracle
    for the finite semantic model, not a learned prediction or a deployment
    feature.  It intentionally does not establish a wall-clock advantage.
    """
    from causal_observation_reproduction.reference.scm_evaluator import (
        SemanticWorldScmEvaluator,
    )

    prototype = SemanticWorldScmEvaluator(dict(semantic_world), seed=scorer_seed)
    remaining_horizon = prototype.horizon if horizon is None else int(horizon)
    if remaining_horizon < 1:
        raise ValueError("signature horizon must be positive")
    action_count = prototype.action_count
    continuation_count = action_count ** max(0, remaining_horizon - 1)
    if continuation_count > max(1, int(max_action_sequences)):
        raise ValueError(
            "finite-horizon action enumeration exceeds the declared scorer-only bound"
        )
    known_variables = set(prototype.variable_ids)
    signatures: list[InterventionalActionSignature] = []
    for context_id, context in sorted(contexts.items()):
        if not context or not set(context) <= known_variables:
            raise ValueError(
                "each signature context must be a nonempty declared state assignment"
            )
        if any(not isfinite(float(value)) for value in context.values()):
            raise ValueError("signature contexts must contain finite numeric values")
        action_values = tuple(
            _best_return_after_first_action(
                semantic_world=dict(semantic_world),
                context=context,
                first_action=action,
                horizon=remaining_horizon,
                action_count=action_count,
                scorer_seed=scorer_seed,
            )
            for action in range(action_count)
        )
        signatures.append(
            InterventionalActionSignature(
                context_id=str(context_id),
                horizon=remaining_horizon,
                action_values=tuple(round(value, 12) for value in action_values),
            )
        )
    if not signatures:
        raise ValueError("at least one declared context is required")
    return ReferenceQuotient(signatures=tuple(signatures), scorer_seed=int(scorer_seed))


def classify_declared_quotient_regime(
    *,
    declared_manifest: Mapping[str, Any],
    reference_quotient: ReferenceQuotient,
) -> DeclaredQuotientRegime:
    """Fail closed when a declared regime disagrees with oracle signatures."""
    declared = str(declared_manifest.get("declared_regime") or "")
    if declared not in _DECLARED_REGIMES:
        raise ValueError("declared_regime must be one of positive, null, or invalid")
    proposed = {
        str(context_id): str(class_id)
        for context_id, class_id in dict(
            declared_manifest.get("proposed_observation_classes") or {}
        ).items()
    }
    expected_contexts = set(reference_quotient.labels_by_context)
    if set(proposed) != expected_contexts:
        raise ValueError(
            "declared quotient classes must cover exactly the oracle contexts"
        )
    grouped: dict[str, set[str]] = {}
    for context_id, class_id in proposed.items():
        grouped.setdefault(class_id, set()).add(
            reference_quotient.labels_by_context[context_id]
        )
    merges_distinct_signatures = any(len(labels) > 1 for labels in grouped.values())
    reference_class_count = len(set(reference_quotient.labels_by_context.values()))
    resolved = (
        "invalid"
        if merges_distinct_signatures
        else "null"
        if reference_class_count == 1
        else "positive"
    )
    if declared != resolved:
        raise ValueError(
            f"declared quotient regime {declared!r} disagrees with oracle-resolved regime {resolved!r}"
        )
    return DeclaredQuotientRegime(
        declared_regime=declared,
        resolved_regime=resolved,
        proposed_classes=proposed,
        reference_quotient=reference_quotient,
    )


def build_generated_poscm_reference_quotient_sidecar(
    bridge: Mapping[str, Any],
    *,
    contexts_by_world: Mapping[str, Mapping[str, Mapping[str, float]]],
    manifests_by_world: Mapping[str, Mapping[str, Any]],
    horizon: int | None = None,
    scorer_seed: int = 0,
    max_action_sequences: int = 4096,
) -> dict[str, Any]:
    """Build a separate scorer artifact for a generated POSCM bridge.

    The bridge itself is intentionally left unchanged.  Keeping oracle action
    signatures in a sidecar makes it mechanically harder for training or
    model-selection code to consume a latent label as an observation feature.
    """
    worlds = [dict(row) for row in bridge.get("worlds") or []]
    world_ids = {str(world.get("world_id") or "") for world in worlds}
    if not world_ids or "" in world_ids:
        raise ValueError("generated POSCM bridge needs uniquely named worlds")
    if set(contexts_by_world) != world_ids or set(manifests_by_world) != world_ids:
        raise ValueError(
            "contexts and declared manifests must cover exactly the generated bridge worlds"
        )
    labels: dict[str, dict[str, Any]] = {}
    for world in worlds:
        world_id = str(world["world_id"])
        semantic_world = dict(
            world.get("semantic_world")
            or dict(world.get("poscm") or {}).get("semantic_world")
            or {}
        )
        if not semantic_world:
            raise ValueError(
                f"generated bridge world {world_id!r} lacks a semantic world"
            )
        reference = derive_finite_horizon_action_signatures(
            semantic_world,
            contexts=contexts_by_world[world_id],
            horizon=horizon,
            scorer_seed=scorer_seed,
            max_action_sequences=max_action_sequences,
        )
        labels[world_id] = classify_declared_quotient_regime(
            declared_manifest=manifests_by_world[world_id],
            reference_quotient=reference,
        ).scorer_payload()
    return {
        "schema_version": INTERVENTIONAL_ACTION_SIGNATURE_SCHEMA_VERSION,
        "artifact_scope": "scorer_only_oracle",
        "selection_eligible": False,
        "deployment_visible": False,
        "source_bridge_id": str(bridge.get("bridge_id") or ""),
        "world_reference_quotients": labels,
    }


def _best_return_after_first_action(
    *,
    semantic_world: dict[str, Any],
    context: Mapping[str, float],
    first_action: int,
    horizon: int,
    action_count: int,
    scorer_seed: int,
) -> float:
    continuations = product(range(action_count), repeat=max(0, horizon - 1))
    return max(
        _paired_rollout_return(
            semantic_world=semantic_world,
            context=context,
            actions=(first_action, *continuation),
            scorer_seed=scorer_seed,
        )
        for continuation in continuations
    )


def _paired_rollout_return(
    *,
    semantic_world: dict[str, Any],
    context: Mapping[str, float],
    actions: Sequence[int],
    scorer_seed: int,
) -> float:
    from causal_observation_reproduction.reference.scm_evaluator import (
        SemanticWorldScmEvaluator,
    )

    evaluator = SemanticWorldScmEvaluator(semantic_world, seed=scorer_seed)
    evaluator.reset(seed=scorer_seed)
    evaluator.state.update({str(key): float(value) for key, value in context.items()})
    evaluator.contextual_target_state = dict(context)
    evaluator.action_history = []
    evaluator.step_index = 0
    total_return = 0.0
    for action in actions:
        _, reward, terminated, truncated, _ = evaluator.step(int(action))
        total_return += float(reward)
        if terminated or truncated:
            break
    return total_return


__all__ = [
    "INTERVENTIONAL_ACTION_SIGNATURE_SCHEMA_VERSION",
    "DeclaredQuotientRegime",
    "InterventionalActionSignature",
    "ReferenceQuotient",
    "build_generated_poscm_reference_quotient_sidecar",
    "classify_declared_quotient_regime",
    "derive_finite_horizon_action_signatures",
    "enumerate_binary_contexts",
]
