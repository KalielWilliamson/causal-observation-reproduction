from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from causal_observation_reproduction.reference.artifacts import (
    artifact_payload_hash,
    payload_value,
)
from causal_observation_reproduction.reference.scm_complexity import (
    validate_structural_equation_spec,
)

SEMANTIC_WORLD_SCHEMA_VERSION = "beb.generated_poscm_semantic_world.v0"


@dataclass(frozen=True)
class StateVariableSpec:
    variable_id: str
    role: str
    observed: bool
    domain: str = "binary"

    def as_payload(self) -> dict[str, Any]:
        return {
            "variable_id": str(self.variable_id),
            "role": str(self.role),
            "domain": str(self.domain),
            "observed": bool(self.observed),
        }


@dataclass(frozen=True)
class ActionSpaceSpec:
    action_variables: tuple[str, ...]
    branching_factor: int
    action_effect_matrix: tuple[tuple[float, ...], ...] | None = None
    domain: str = "discrete"

    def as_payload(self) -> dict[str, Any]:
        action_variables = [str(item) for item in self.action_variables]
        matrix = self.action_effect_matrix or _default_action_effect_matrix(
            len(action_variables)
        )
        return {
            "action_variables": action_variables,
            "action_count": len(action_variables),
            "branching_factor": max(1, int(self.branching_factor)),
            "action_effect_matrix": [[float(value) for value in row] for row in matrix],
            "domain": str(self.domain),
        }


@dataclass(frozen=True)
class TransitionModelRowSpec:
    variable_id: str
    parents: tuple[str, ...]
    action_inputs: tuple[str, ...]
    latent_noise_inputs: tuple[str, ...]
    equation: str
    structural_equation: dict[str, Any]
    transition_delay: int = 0

    def as_payload(self) -> dict[str, Any]:
        return {
            "variable_id": str(self.variable_id),
            "parents": [str(item) for item in self.parents],
            "action_inputs": [str(item) for item in self.action_inputs],
            "latent_noise_inputs": [str(item) for item in self.latent_noise_inputs],
            "equation": str(self.equation),
            "structural_equation": dict(self.structural_equation),
            "transition_delay": max(0, int(self.transition_delay)),
        }


@dataclass(frozen=True)
class RewardModelSpec:
    reward_id: str
    outcome_variables: tuple[str, ...]
    action_variables: tuple[str, ...]
    reward_delay: int
    success_reward: float
    failure_reward: float
    success_condition: str
    credit_assignment_target: str
    success_threshold: float = 0.60

    def as_payload(self) -> dict[str, Any]:
        return {
            "reward_id": str(self.reward_id),
            "outcome_variables": [str(item) for item in self.outcome_variables],
            "action_variables": [str(item) for item in self.action_variables],
            "reward_delay": max(0, int(self.reward_delay)),
            "success_reward": float(self.success_reward),
            "failure_reward": float(self.failure_reward),
            "success_condition": str(self.success_condition),
            "credit_assignment_target": str(self.credit_assignment_target),
            "success_threshold": _clamp_unit(float(self.success_threshold)),
        }


@dataclass(frozen=True)
class ObservationModelSpec:
    observed_variables: tuple[str, ...]
    latent_variables: tuple[str, ...]
    observation_noise: float
    proxy_reliability: float
    spurious_correlation_strength: float
    event_emitters: dict[str, list[str]]
    emission_policy: str = "emit_observed_state_and_configured_event_roles"

    def as_payload(self) -> dict[str, Any]:
        return {
            "observed_variables": [str(item) for item in self.observed_variables],
            "latent_variables": [str(item) for item in self.latent_variables],
            "observation_noise": _clamp_unit(float(self.observation_noise)),
            "proxy_reliability": _clamp_unit(float(self.proxy_reliability)),
            "spurious_correlation_strength": _clamp_unit(
                float(self.spurious_correlation_strength)
            ),
            "event_emitters": {
                str(role): [str(item) for item in variables]
                for role, variables in self.event_emitters.items()
            },
            "emission_policy": str(self.emission_policy),
        }


@dataclass(frozen=True)
class InterventionSpec:
    intervention_id: str
    intervention_type: str
    targets: tuple[Any, ...]
    parameters: dict[str, Any]

    def as_payload(self) -> dict[str, Any]:
        return {
            "intervention_id": str(self.intervention_id),
            "intervention_type": str(self.intervention_type),
            "targets": payload_value(self.targets),
            "parameters": payload_value(self.parameters),
        }


@dataclass(frozen=True)
class SemanticWorldSpec:
    state_variables: tuple[StateVariableSpec, ...]
    action_space: ActionSpaceSpec
    transition_model: tuple[TransitionModelRowSpec, ...]
    reward_model: RewardModelSpec
    observation_model: ObservationModelSpec
    interventions: tuple[InterventionSpec, ...]
    difficulty_vector: dict[str, Any]
    complexity: dict[str, Any]
    target_region: str
    index: int

    @property
    def semantic_world_id(self) -> str:
        return (
            "generated-poscm-semantic-world-"
            + artifact_payload_hash(self._identity_basis())[:16]
        )

    def as_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SEMANTIC_WORLD_SCHEMA_VERSION,
            "semantic_world_id": self.semantic_world_id,
            "state_variables": [item.as_payload() for item in self.state_variables],
            "action_space": self.action_space.as_payload(),
            "transition_model": [item.as_payload() for item in self.transition_model],
            "reward_model": self.reward_model.as_payload(),
            "observation_model": self.observation_model.as_payload(),
            "interventions": [item.as_payload() for item in self.interventions],
            "difficulty_vector": payload_value(self.difficulty_vector),
            "complexity": payload_value(self.complexity),
        }

    def _identity_basis(self) -> dict[str, Any]:
        return {
            "node_ids": [item.variable_id for item in self.state_variables],
            "action_variables": list(self.action_space.action_variables),
            "transition_model": [item.as_payload() for item in self.transition_model],
            "observation_model": self.observation_model.as_payload(),
            "reward_model": self.reward_model.as_payload(),
            "difficulty_vector": payload_value(self.difficulty_vector),
            "complexity": payload_value(self.complexity),
        }


def validate_semantic_world_payload(payload: dict[str, Any]) -> dict[str, bool]:
    action_space = dict(payload.get("action_space") or {})
    transition_model = [dict(row) for row in payload.get("transition_model") or []]
    reward_model = dict(payload.get("reward_model") or {})
    observation_model = dict(payload.get("observation_model") or {})
    difficulty = dict(payload.get("difficulty_vector") or {})
    complexity = dict(payload.get("complexity") or {})
    action_variables = [
        str(item) for item in action_space.get("action_variables") or []
    ]
    outcome_variables = [
        str(item) for item in reward_model.get("outcome_variables") or []
    ]
    return {
        "schema_version": payload.get("schema_version")
        == SEMANTIC_WORLD_SCHEMA_VERSION,
        "has_semantic_world_id": bool(str(payload.get("semantic_world_id") or "")),
        "has_state_variables": bool(payload.get("state_variables")),
        "has_action_space": bool(action_variables)
        and int(action_space.get("action_count") or 0) == len(action_variables),
        "has_action_effect_matrix": _valid_action_effect_matrix(
            matrix=action_space.get("action_effect_matrix"),
            action_count=len(action_variables),
            action_variable_count=len(action_variables),
        ),
        "has_transition_model": bool(transition_model),
        "has_reward_model": bool(reward_model),
        "reward_not_constant": float(reward_model.get("success_reward") or 0.0)
        > float(reward_model.get("failure_reward") or 0.0),
        "success_threshold_valid": 0.0
        <= float(reward_model.get("success_threshold") or 0.5)
        <= 1.0,
        "action_can_reach_outcome": _semantic_action_can_reach_outcome(
            action_variables=action_variables,
            outcome_variables=outcome_variables,
            transition_model=transition_model,
        ),
        "has_observation_model": bool(observation_model),
        "observations_nontrivial": bool(observation_model.get("observed_variables"))
        and bool(observation_model.get("latent_variables")),
        "has_interventions": bool(payload.get("interventions")),
        "difficulty_vector_present": _difficulty_vector_present(difficulty),
        "has_complexity": bool(complexity),
        "structural_equations_valid": all(
            all(
                validate_structural_equation_spec(
                    dict(row.get("structural_equation") or {})
                ).values()
            )
            for row in transition_model
        ),
    }


def _semantic_action_can_reach_outcome(
    *,
    action_variables: list[str],
    outcome_variables: list[str],
    transition_model: list[dict[str, Any]],
) -> bool:
    children: dict[str, list[str]] = {}
    for row in transition_model:
        target = str(row.get("variable_id") or "")
        for parent in [
            *list(row.get("parents") or []),
            *list(row.get("action_inputs") or []),
        ]:
            children.setdefault(str(parent), []).append(target)
    for action in action_variables:
        frontier = [action]
        seen: set[str] = set()
        while frontier:
            node = frontier.pop(0)
            if node in seen:
                continue
            seen.add(node)
            if node in set(outcome_variables):
                return True
            frontier.extend(children.get(node, []))
    return False


def _difficulty_vector_present(difficulty: dict[str, Any]) -> bool:
    required = {
        "horizon",
        "branching_factor",
        "action_count",
        "reward_delay",
        "observation_noise",
        "latent_confounding_strength",
        "proxy_reliability",
        "spurious_correlation_strength",
        "topology_edge_density",
        "latent_fraction",
        "complexity_tier",
        "interaction_order",
        "nonlinear_transform_count",
        "temporal_lag",
    }
    return required <= set(difficulty)


def _default_action_effect_matrix(width: int) -> tuple[tuple[float, ...], ...]:
    size = max(1, int(width))
    return tuple(
        tuple(1.0 if row == col else -1.0 for col in range(size)) for row in range(size)
    )


def _valid_action_effect_matrix(
    *,
    matrix: Any,
    action_count: int,
    action_variable_count: int,
) -> bool:
    if not isinstance(matrix, list) or len(matrix) != int(action_count):
        return False
    for row in matrix:
        if not isinstance(row, list) or len(row) != int(action_variable_count):
            return False
        try:
            [float(value) for value in row]
        except (TypeError, ValueError):
            return False
    return True


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
