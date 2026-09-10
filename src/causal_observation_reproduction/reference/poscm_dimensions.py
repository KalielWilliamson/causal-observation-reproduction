from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from causal_observation_reproduction.reference.numeric import clamp01

POSCM_STRUCTURAL_DIMENSION_SCHEMA_VERSION = (
    "causal_observation_reproduction.poscm_structural_dimensions.v0"
)
POSCM_OPERATING_ENVELOPE_SCHEMA_VERSION = (
    "causal_observation_reproduction.poscm_operating_envelope.v0"
)

POSCM_STRUCTURAL_DIMENSION_NAMES = (
    "solution_density_target",
    "gate_depth",
    "action_branching_factor",
    "correct_action_fraction",
    "distractor_count",
    "distractor_depth",
    "trap_strength",
    "proxy_strength",
    "proxy_observability",
    "collider_strength",
    "delayed_penalty_strength",
    "latent_confounding_strength",
    "noise_scale",
    "reward_delay",
    "observation_mask_fraction",
)


@dataclass(frozen=True)
class PoscmDimensionBound:
    name: str
    minimum: float
    maximum: float
    rationale: str = ""
    log_scale: bool = False

    def contains(self, value: float) -> bool:
        return float(self.minimum) <= float(value) <= float(self.maximum)

    def normalized_edge_distance(self, value: float) -> float:
        if self.log_scale:
            if (
                float(value) <= 0.0
                or float(self.minimum) <= 0.0
                or float(self.maximum) <= 0.0
            ):
                return 0.0
            log_minimum = math.log10(float(self.minimum))
            log_maximum = math.log10(float(self.maximum))
            log_value = math.log10(float(value))
            width = max(1e-12, log_maximum - log_minimum)
            lower = (log_value - log_minimum) / width
            upper = (log_maximum - log_value) / width
            return min(lower, upper)
        width = max(1e-12, float(self.maximum) - float(self.minimum))
        lower = (float(value) - float(self.minimum)) / width
        upper = (float(self.maximum) - float(value)) / width
        return min(lower, upper)

    def as_payload(self) -> dict[str, Any]:
        return {
            "name": str(self.name),
            "minimum": float(self.minimum),
            "maximum": float(self.maximum),
            "rationale": str(self.rationale),
            "log_scale": bool(self.log_scale),
        }


DEFAULT_POSCM_OPERATING_ENVELOPE_BOUNDS = (
    PoscmDimensionBound(
        "solution_density_target",
        1e-8,
        1.0,
        "Covers sparse through trivial policy-solution support.",
        log_scale=True,
    ),
    PoscmDimensionBound(
        "gate_depth",
        1.0,
        12.0,
        "Covers short local chains through long-horizon agent-memory credit assignment.",
    ),
    PoscmDimensionBound(
        "action_branching_factor",
        2.0,
        16.0,
        "Covers binary controls through high but still experimentable action branching.",
    ),
    PoscmDimensionBound(
        "correct_action_fraction",
        1.0 / 16.0,
        1.0,
        "Matches the supported action-branching envelope.",
    ),
    PoscmDimensionBound(
        "distractor_count",
        0.0,
        15.0,
        "Allows every non-correct action to be a distractor at maximum branching.",
    ),
    PoscmDimensionBound(
        "distractor_depth", 0.0, 8.0, "Covers shallow and multi-step trap/proxy paths."
    ),
    PoscmDimensionBound("trap_strength", 0.0, 1.0, "Unit-scaled trap semantics."),
    PoscmDimensionBound(
        "proxy_strength", 0.0, 1.0, "Unit-scaled spurious proxy strength."
    ),
    PoscmDimensionBound(
        "proxy_observability", 0.0, 1.0, "Unit-scaled proxy visibility."
    ),
    PoscmDimensionBound(
        "collider_strength", 0.0, 1.0, "Unit-scaled collider-trap strength."
    ),
    PoscmDimensionBound(
        "delayed_penalty_strength", 0.0, 1.0, "Unit-scaled delayed penalty strength."
    ),
    PoscmDimensionBound(
        "latent_confounding_strength", 0.0, 1.0, "Unit-scaled latent confounding."
    ),
    PoscmDimensionBound(
        "noise_scale", 0.0, 1.0, "Unit-scaled observation and transition noise."
    ),
    PoscmDimensionBound(
        "reward_delay",
        0.0,
        16.0,
        "Covers immediate through long delayed reward assignment.",
    ),
    PoscmDimensionBound(
        "observation_mask_fraction", 0.0, 1.0, "Unit-scaled partial observability."
    ),
)


@dataclass(frozen=True)
class PoscmEnvelopeValidation:
    envelope_id: str
    status: str
    violations: tuple[dict[str, Any], ...]
    boundary_dimensions: tuple[str, ...]
    boundary_margin_fraction: float

    @property
    def in_envelope(self) -> bool:
        return self.status in {"inside_envelope", "boundary_probe"}

    def as_payload(self) -> dict[str, Any]:
        return {
            "schema_version": POSCM_OPERATING_ENVELOPE_SCHEMA_VERSION,
            "envelope_id": str(self.envelope_id),
            "status": str(self.status),
            "in_envelope": bool(self.in_envelope),
            "violations": [dict(item) for item in self.violations],
            "boundary_dimensions": list(self.boundary_dimensions),
            "boundary_margin_fraction": float(self.boundary_margin_fraction),
        }


@dataclass(frozen=True)
class PoscmOperatingEnvelope:
    envelope_id: str = "paper1_agent_memory_control_v0"
    description: str = (
        "Bounded POSCM structural envelope for Paper 1 agent-memory/control experiments. "
        "The envelope is intended to cover the claim-relevant real-world cases while making "
        "out-of-envelope probes explicit."
    )
    bounds: tuple[PoscmDimensionBound, ...] = DEFAULT_POSCM_OPERATING_ENVELOPE_BOUNDS
    boundary_margin_fraction: float = 0.05

    def bounds_by_name(self) -> dict[str, PoscmDimensionBound]:
        return {bound.name: bound for bound in self.bounds}

    def validate(
        self, dimensions: PoscmStructuralDimensions
    ) -> PoscmEnvelopeValidation:
        violations: list[dict[str, Any]] = []
        boundary_dimensions: list[str] = []
        for bound in self.bounds:
            value = float(getattr(dimensions, bound.name))
            if not bound.contains(value):
                violations.append(
                    {
                        "dimension": str(bound.name),
                        "value": value,
                        "minimum": float(bound.minimum),
                        "maximum": float(bound.maximum),
                    }
                )
            elif bound.normalized_edge_distance(value) <= float(
                self.boundary_margin_fraction
            ):
                boundary_dimensions.append(str(bound.name))
        if violations:
            status = "out_of_envelope"
        elif boundary_dimensions:
            status = "boundary_probe"
        else:
            status = "inside_envelope"
        return PoscmEnvelopeValidation(
            envelope_id=str(self.envelope_id),
            status=status,
            violations=tuple(violations),
            boundary_dimensions=tuple(boundary_dimensions),
            boundary_margin_fraction=float(self.boundary_margin_fraction),
        )

    def as_payload(self) -> dict[str, Any]:
        return {
            "schema_version": POSCM_OPERATING_ENVELOPE_SCHEMA_VERSION,
            "envelope_id": str(self.envelope_id),
            "description": str(self.description),
            "boundary_margin_fraction": float(self.boundary_margin_fraction),
            "bounds": [bound.as_payload() for bound in self.bounds],
        }


DEFAULT_POSCM_OPERATING_ENVELOPE = PoscmOperatingEnvelope()


@dataclass(frozen=True)
class PoscmStructuralDimensions:
    solution_density_target: float = 0.15
    gate_depth: float = 3.0
    action_branching_factor: float = 2.0
    correct_action_fraction: float = 0.5
    distractor_count: float = 0.0
    distractor_depth: float = 1.0
    trap_strength: float = 0.0
    proxy_strength: float = 0.0
    proxy_observability: float = 1.0
    collider_strength: float = 0.0
    delayed_penalty_strength: float = 0.0
    latent_confounding_strength: float = 0.30
    noise_scale: float = 0.10
    reward_delay: float = 1.0
    observation_mask_fraction: float = 0.75

    @classmethod
    def from_ladder_controls(
        cls,
        *,
        action_count: int = 3,
        gate_depth: int = 4,
        distractor_count: int = 2,
        target_solution_density: float | None = None,
        distractor_depth: float = 1.0,
        trap_strength: float = 0.75,
        proxy_strength: float = 0.8,
        proxy_observability: float = 1.0,
        collider_strength: float = 0.6,
        delayed_penalty_strength: float = 0.75,
        latent_confounding_strength: float = 0.3,
        noise_scale: float = 0.1,
        reward_delay: float = 1.0,
        observation_mask_fraction: float = 0.5,
    ) -> PoscmStructuralDimensions:
        action_count = max(2, int(action_count))
        gate_depth = max(1, int(gate_depth))
        exact_density = 1.0 / float(action_count**gate_depth)
        return cls(
            solution_density_target=float(target_solution_density)
            if target_solution_density is not None
            else exact_density,
            gate_depth=float(gate_depth),
            action_branching_factor=float(action_count),
            correct_action_fraction=1.0 / float(action_count),
            distractor_count=float(
                min(action_count - 1, max(0, int(distractor_count)))
            ),
            distractor_depth=float(distractor_depth),
            trap_strength=clamp01(trap_strength),
            proxy_strength=clamp01(proxy_strength),
            proxy_observability=clamp01(proxy_observability),
            collider_strength=clamp01(collider_strength),
            delayed_penalty_strength=clamp01(delayed_penalty_strength),
            latent_confounding_strength=clamp01(latent_confounding_strength),
            noise_scale=max(0.0, float(noise_scale)),
            reward_delay=float(max(0.0, reward_delay)),
            observation_mask_fraction=clamp01(observation_mask_fraction),
        )

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> PoscmStructuralDimensions:
        defaults = cls()
        values = {
            column: _finite_float(payload.get(column), float(getattr(defaults, column)))
            for column in POSCM_STRUCTURAL_DIMENSION_NAMES
        }
        if "action_count" in payload and "action_branching_factor" not in payload:
            action_count = max(2, int(payload.get("action_count") or 2))
            values["action_branching_factor"] = float(action_count)
        if "correct_action_fraction" not in payload:
            action_count = max(2, int(round(float(values["action_branching_factor"]))))
            values["correct_action_fraction"] = 1.0 / float(action_count)
        return cls(**values)

    @classmethod
    def from_motif_design_matrix_row(cls, row: Any) -> PoscmStructuralDimensions:
        array = np.asarray(row, dtype=float).reshape(-1)
        defaults = cls()
        values = {
            column: float(array[index])
            if index < len(array) and np.isfinite(float(array[index]))
            else float(getattr(defaults, column))
            for index, column in enumerate(POSCM_STRUCTURAL_DIMENSION_NAMES)
        }
        return cls(**values)

    @property
    def action_count(self) -> int:
        return max(2, int(round(float(self.action_branching_factor))))

    @property
    def gate_depth_count(self) -> int:
        return max(1, int(round(float(self.gate_depth))))

    @property
    def distractor_count_int(self) -> int:
        return min(
            self.action_count - 1, max(0, int(round(float(self.distractor_count))))
        )

    @property
    def exact_solution_density(self) -> float:
        return 1.0 / float(self.action_count**self.gate_depth_count)

    @property
    def derived_region_label(self) -> str:
        return summarize_poscm_structural_region(self)

    @property
    def hardness_proxy_score(self) -> float:
        sparsity = -math.log10(max(1e-12, float(self.exact_solution_density)))
        traps = float(self.distractor_count_int) * (0.5 + float(self.trap_strength))
        observability = 1.0 - clamp01(self.observation_mask_fraction)
        proxy = 0.5 * float(self.proxy_strength) + 0.5 * float(self.collider_strength)
        confounding = float(self.latent_confounding_strength) + float(self.noise_scale)
        delay = 0.15 * float(max(0.0, self.reward_delay))
        return float(sparsity + traps + observability + proxy + confounding + delay)

    def dimension_payload(self) -> dict[str, float]:
        return {
            column: float(getattr(self, column))
            for column in POSCM_STRUCTURAL_DIMENSION_NAMES
        }

    def as_payload(self) -> dict[str, Any]:
        return {
            "schema_version": POSCM_STRUCTURAL_DIMENSION_SCHEMA_VERSION,
            **self.dimension_payload(),
            "action_count": int(self.action_count),
            "gate_depth_count": int(self.gate_depth_count),
            "distractor_count_int": int(self.distractor_count_int),
            "exact_solution_density": float(self.exact_solution_density),
            "derived_region_label": self.derived_region_label,
            "hardness_proxy_score": float(self.hardness_proxy_score),
        }

    def to_motif_design_matrix(self) -> np.ndarray:
        return np.asarray(
            [
                [
                    float(getattr(self, column))
                    for column in POSCM_STRUCTURAL_DIMENSION_NAMES
                ]
            ],
            dtype=float,
        )

    def same_dimensions_as(
        self, other: PoscmStructuralDimensions, *, tolerance: float = 1e-12
    ) -> bool:
        return all(
            abs(float(getattr(self, column)) - float(getattr(other, column)))
            <= float(tolerance)
            for column in POSCM_STRUCTURAL_DIMENSION_NAMES
        )

    def envelope_validation(
        self,
        envelope: PoscmOperatingEnvelope = DEFAULT_POSCM_OPERATING_ENVELOPE,
    ) -> PoscmEnvelopeValidation:
        return envelope.validate(self)


def poscm_structural_dimensions_from_ladder_controls(
    *,
    action_count: int = 3,
    gate_depth: int = 4,
    distractor_count: int = 2,
    target_solution_density: float | None = None,
    **kwargs: Any,
) -> PoscmStructuralDimensions:
    return PoscmStructuralDimensions.from_ladder_controls(
        action_count=action_count,
        gate_depth=gate_depth,
        distractor_count=distractor_count,
        target_solution_density=target_solution_density,
        **kwargs,
    )


def poscm_structural_dimensions_from_payload(
    payload: dict[str, Any],
) -> PoscmStructuralDimensions:
    return PoscmStructuralDimensions.from_payload(payload)


def suggest_poscm_structural_dimensions(
    trial: Any,
    *,
    action_counts: tuple[int, ...] = (2, 3, 4),
    gate_depths: tuple[int, ...] = (2, 3, 4, 5),
    max_reward_delay: int = 3,
) -> PoscmStructuralDimensions:
    action_axis = tuple(max(2, int(item)) for item in action_counts) or (2,)
    gate_axis = tuple(max(1, int(item)) for item in gate_depths) or (1,)
    action_count = int(_suggest_categorical(trial, "action_count", action_axis))
    gate_depth = int(_suggest_categorical(trial, "gate_depth", gate_axis))
    raw_distractor_count = int(
        _suggest_int(trial, "distractor_count", 0, max(0, max(action_axis) - 1))
    )
    distractor_count = min(max(0, action_count - 1), raw_distractor_count)
    exact_density = 1.0 / float(action_count**gate_depth)
    density_scale = float(_suggest_float(trial, "solution_density_scale", 0.75, 1.25))
    return PoscmStructuralDimensions.from_ladder_controls(
        action_count=action_count,
        gate_depth=gate_depth,
        distractor_count=distractor_count,
        target_solution_density=min(1.0, max(1e-12, exact_density * density_scale)),
        distractor_depth=float(_suggest_float(trial, "distractor_depth", 1.0, 3.0)),
        trap_strength=float(_suggest_float(trial, "trap_strength", 0.2, 1.0)),
        proxy_strength=float(_suggest_float(trial, "proxy_strength", 0.0, 1.0)),
        proxy_observability=float(
            _suggest_float(trial, "proxy_observability", 0.35, 1.0)
        ),
        collider_strength=float(_suggest_float(trial, "collider_strength", 0.0, 1.0)),
        delayed_penalty_strength=float(
            _suggest_float(trial, "delayed_penalty_strength", 0.0, 1.0)
        ),
        latent_confounding_strength=float(
            _suggest_float(trial, "latent_confounding_strength", 0.0, 0.8)
        ),
        noise_scale=float(_suggest_float(trial, "noise_scale", 0.0, 0.35)),
        reward_delay=float(
            _suggest_int(trial, "reward_delay", 0, max(0, int(max_reward_delay)))
        ),
        observation_mask_fraction=float(
            _suggest_float(trial, "observation_mask_fraction", 0.0, 0.85)
        ),
    )


def summarize_poscm_structural_region(vector: PoscmStructuralDimensions) -> str:
    if float(vector.trap_strength) > 0.0 and float(vector.proxy_strength) > 0.0:
        if float(vector.collider_strength) > 0.0:
            return "spurious_proxy_collider_trap"
        return "spurious_proxy_trap"
    if float(vector.gate_depth) >= 3.0:
        return "sparse_chain"
    return "baseline_topology"


def _suggest_categorical(trial: Any, name: str, choices: tuple[Any, ...]) -> Any:
    return trial.suggest_categorical(name, list(dict.fromkeys(choices)))


def _suggest_float(trial: Any, name: str, low: float, high: float) -> float:
    return float(trial.suggest_float(name, float(low), float(high)))


def _suggest_int(trial: Any, name: str, low: int, high: int) -> int:
    if hasattr(trial, "suggest_int"):
        return int(trial.suggest_int(name, int(low), int(high)))
    return int(_suggest_categorical(trial, name, tuple(range(int(low), int(high) + 1))))


def _finite_float(value: Any, default: float) -> float:
    try:
        selected = float(value)
    except (TypeError, ValueError):
        return float(default)
    return selected if np.isfinite(selected) else float(default)


__all__ = [
    "DEFAULT_POSCM_OPERATING_ENVELOPE",
    "DEFAULT_POSCM_OPERATING_ENVELOPE_BOUNDS",
    "POSCM_OPERATING_ENVELOPE_SCHEMA_VERSION",
    "POSCM_STRUCTURAL_DIMENSION_NAMES",
    "POSCM_STRUCTURAL_DIMENSION_SCHEMA_VERSION",
    "PoscmDimensionBound",
    "PoscmEnvelopeValidation",
    "PoscmOperatingEnvelope",
    "PoscmStructuralDimensions",
    "poscm_structural_dimensions_from_ladder_controls",
    "poscm_structural_dimensions_from_payload",
    "suggest_poscm_structural_dimensions",
    "summarize_poscm_structural_region",
]
