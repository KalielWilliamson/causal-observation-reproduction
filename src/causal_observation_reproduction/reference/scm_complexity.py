from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, cast

import networkx as nx
import numpy as np
import sympy as sp
from sympy.polys.polyerrors import GeneratorsNeeded, PolynomialError

from causal_observation_reproduction.reference.artifacts import (
    artifact_payload_hash,
    payload_value,
)
from causal_observation_reproduction.reference.domain import CausalGraphSpec
from causal_observation_reproduction.reference.numeric import clamp01

SCM_COMPLEXITY_SCHEMA_VERSION = "beb.generated_poscm_scm_complexity.v0"
STRUCTURAL_EQUATION_SCHEMA_VERSION = "beb.generated_poscm_structural_equation.v0"
SCM_COMPLEXITY_VECTOR_FIELDS = (
    "polynomial_order",
    "interaction_strength",
    "nonlinear_strength",
    "latent_confounding_strength",
    "temporal_memory_strength",
    "partial_observability",
    "noise_scale",
    "hierarchy_depth",
)


class ScmComplexityTier(StrEnum):
    TIER_0_LINEAR = "tier_0_linear"
    TIER_1_INTERACTIONS = "tier_1_interactions"
    TIER_2_SMOOTH_NONLINEAR = "tier_2_smooth_nonlinear"
    TIER_3_LATENT_CONFOUNDING = "tier_3_latent_confounding"
    TIER_4_TEMPORAL = "tier_4_temporal"
    TIER_5_COMPOSITIONAL = "tier_5_compositional"


COMPLEXITY_TIER_ORDER = {
    ScmComplexityTier.TIER_0_LINEAR: 0,
    ScmComplexityTier.TIER_1_INTERACTIONS: 1,
    ScmComplexityTier.TIER_2_SMOOTH_NONLINEAR: 2,
    ScmComplexityTier.TIER_3_LATENT_CONFOUNDING: 3,
    ScmComplexityTier.TIER_4_TEMPORAL: 4,
    ScmComplexityTier.TIER_5_COMPOSITIONAL: 5,
}


@dataclass(frozen=True)
class ScmComplexityVector:
    polynomial_order: float = 1.0
    interaction_strength: float = 0.0
    nonlinear_strength: float = 0.0
    latent_confounding_strength: float = 0.0
    temporal_memory_strength: float = 0.0
    partial_observability: float = 0.0
    noise_scale: float = 0.0
    hierarchy_depth: float = 0.0

    def as_payload(self) -> dict[str, float]:
        return {
            "polynomial_order": float(max(1.0, self.polynomial_order)),
            "interaction_strength": _clamp_nonnegative(self.interaction_strength),
            "nonlinear_strength": _clamp_nonnegative(self.nonlinear_strength),
            "latent_confounding_strength": _clamp_nonnegative(
                self.latent_confounding_strength
            ),
            "temporal_memory_strength": _clamp_nonnegative(
                self.temporal_memory_strength
            ),
            "partial_observability": _clamp_nonnegative(self.partial_observability),
            "noise_scale": _clamp_nonnegative(self.noise_scale),
            "hierarchy_depth": _clamp_nonnegative(self.hierarchy_depth),
        }


@dataclass(frozen=True)
class StructuralEquationConfig:
    complexity_tier: str = ScmComplexityTier.TIER_0_LINEAR.value
    interaction_order: int = 1
    nonlinear_transform_count: int = 0
    temporal_lag: int = 0
    threshold_count: int = 0
    complexity_vector: ScmComplexityVector | None = None

    def as_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["complexity_vector"] = resolve_complexity_vector(self).as_payload()
        return cast("dict[str, Any]", payload_value(payload))


@dataclass(frozen=True)
class StructuralTermSpec:
    term_type: str
    coefficient: float
    variables: tuple[str, ...]
    powers: tuple[int, ...]

    def as_payload(self) -> dict[str, Any]:
        return {
            "term_type": str(self.term_type),
            "coefficient": float(self.coefficient),
            "variables": [str(item) for item in self.variables],
            "powers": [int(item) for item in self.powers],
        }


@dataclass(frozen=True)
class NonlinearTransformSpec:
    transform: str
    variable: str

    def as_payload(self) -> dict[str, Any]:
        return {
            "transform": str(self.transform),
            "variable": str(self.variable),
        }


@dataclass(frozen=True)
class TemporalLagSpec:
    variable: str
    lag: int
    input: str

    def as_payload(self) -> dict[str, Any]:
        return {
            "variable": str(self.variable),
            "lag": max(1, int(self.lag)),
            "input": str(self.input),
        }


@dataclass(frozen=True)
class StructuralEquationMetrics:
    polynomial_degree: int
    input_count: int
    interaction_term_count: int
    nonlinear_transform_count: int
    threshold_count: int
    latent_input_count: int
    temporal_lag_depth: int

    def as_payload(self) -> dict[str, int]:
        return {
            "polynomial_degree": max(0, int(self.polynomial_degree)),
            "input_count": max(0, int(self.input_count)),
            "interaction_term_count": max(0, int(self.interaction_term_count)),
            "nonlinear_transform_count": max(0, int(self.nonlinear_transform_count)),
            "threshold_count": max(0, int(self.threshold_count)),
            "latent_input_count": max(0, int(self.latent_input_count)),
            "temporal_lag_depth": max(0, int(self.temporal_lag_depth)),
        }


@dataclass(frozen=True)
class StructuralEquationSpec:
    equation_id: str
    variable_id: str
    complexity_tier: str
    complexity_vector: ScmComplexityVector
    inputs: tuple[str, ...]
    parent_variables: tuple[str, ...]
    action_inputs: tuple[str, ...]
    latent_noise_inputs: tuple[str, ...]
    terms: tuple[StructuralTermSpec, ...]
    transforms: tuple[NonlinearTransformSpec, ...]
    temporal_lags: tuple[TemporalLagSpec, ...]
    sympy_expression: str
    metrics: StructuralEquationMetrics

    def as_payload(self) -> dict[str, Any]:
        return {
            "schema_version": STRUCTURAL_EQUATION_SCHEMA_VERSION,
            "equation_id": str(self.equation_id),
            "variable_id": str(self.variable_id),
            "complexity_tier": str(self.complexity_tier),
            "complexity_vector": self.complexity_vector.as_payload(),
            "inputs": [str(item) for item in self.inputs],
            "parent_variables": [str(item) for item in self.parent_variables],
            "action_inputs": [str(item) for item in self.action_inputs],
            "latent_noise_inputs": [str(item) for item in self.latent_noise_inputs],
            "terms": [item.as_payload() for item in self.terms],
            "transforms": [item.as_payload() for item in self.transforms],
            "temporal_lags": [item.as_payload() for item in self.temporal_lags],
            "sympy_expression": str(self.sympy_expression),
            "metrics": self.metrics.as_payload(),
        }


def complexity_vector_from_tier(tier: str) -> ScmComplexityVector:
    tier_level = COMPLEXITY_TIER_ORDER[_coerce_tier(tier)]
    return ScmComplexityVector(
        polynomial_order=1.0 if tier_level == 0 else min(3.0, float(tier_level + 1)),
        interaction_strength=0.0 if tier_level < 1 else min(1.0, 0.25 * tier_level),
        nonlinear_strength=0.0 if tier_level < 2 else min(1.0, 0.25 * (tier_level - 1)),
        latent_confounding_strength=0.0
        if tier_level < 3
        else min(1.0, 0.30 * (tier_level - 2)),
        temporal_memory_strength=0.0
        if tier_level < 4
        else min(1.0, 0.35 * (tier_level - 3)),
        partial_observability=0.0
        if tier_level < 3
        else min(1.0, 0.20 * (tier_level - 2)),
        noise_scale=0.05 * tier_level,
        hierarchy_depth=0.0 if tier_level < 5 else 1.0,
    )


def complexity_vector_from_matrix(
    matrix: np.ndarray | list[list[float]] | tuple[tuple[float, ...], ...],
    *,
    row_index: int = 0,
) -> ScmComplexityVector:
    array = np.asarray(matrix, dtype=float)
    if array.ndim == 1:
        vector = array
    elif array.ndim == 2:
        if array.shape[0] == 0:
            vector = np.zeros(len(SCM_COMPLEXITY_VECTOR_FIELDS), dtype=float)
        else:
            vector = array[int(row_index) % int(array.shape[0])]
    else:
        raise ValueError("SCM complexity design matrix must be one- or two-dimensional")
    padded = np.zeros(len(SCM_COMPLEXITY_VECTOR_FIELDS), dtype=float)
    width = min(len(padded), int(vector.shape[0]))
    padded[:width] = vector[:width]
    return ScmComplexityVector(
        **dict(zip(SCM_COMPLEXITY_VECTOR_FIELDS, padded, strict=True))
    )


def resolve_complexity_vector(config: StructuralEquationConfig) -> ScmComplexityVector:
    if config.complexity_vector is not None:
        return config.complexity_vector
    preset = complexity_vector_from_tier(config.complexity_tier).as_payload()
    return ScmComplexityVector(
        polynomial_order=max(
            preset["polynomial_order"], float(max(1, int(config.interaction_order)))
        ),
        interaction_strength=max(
            preset["interaction_strength"],
            _count_to_strength(int(config.interaction_order) - 1),
        ),
        nonlinear_strength=max(
            preset["nonlinear_strength"],
            _count_to_strength(int(config.nonlinear_transform_count)),
        ),
        latent_confounding_strength=preset["latent_confounding_strength"],
        temporal_memory_strength=max(
            preset["temporal_memory_strength"],
            _count_to_strength(int(config.temporal_lag)),
        ),
        partial_observability=preset["partial_observability"],
        noise_scale=preset["noise_scale"],
        hierarchy_depth=preset["hierarchy_depth"],
    )


def build_structural_equation_spec(
    *,
    variable_id: str,
    parent_variables: tuple[str, ...] = (),
    action_inputs: tuple[str, ...] = (),
    latent_noise_inputs: tuple[str, ...] = (),
    config: StructuralEquationConfig | None = None,
) -> dict[str, Any]:
    cfg = config or StructuralEquationConfig()
    tier = _coerce_tier(cfg.complexity_tier)
    complexity_vector = resolve_complexity_vector(cfg)
    inputs = _ordered_unique([*parent_variables, *action_inputs, *latent_noise_inputs])
    expression, terms, transforms, temporal_lags = _build_expression(
        variable_id=variable_id,
        inputs=inputs,
        latent_noise_inputs=tuple(latent_noise_inputs),
        tier=tier,
        cfg=cfg,
        complexity_vector=complexity_vector,
    )
    metrics = structural_equation_metrics(
        expression=expression,
        inputs=tuple(inputs),
        latent_noise_inputs=tuple(latent_noise_inputs),
        temporal_lags=tuple(temporal_lags),
        transforms=tuple(transforms),
        terms=tuple(terms),
    )
    basis = {
        "variable_id": variable_id,
        "tier": tier.value,
        "inputs": inputs,
        "expression": str(expression),
        "metrics": metrics.as_payload(),
        "complexity_vector": complexity_vector.as_payload(),
    }
    spec = StructuralEquationSpec(
        equation_id="generated-scm-equation-" + artifact_payload_hash(basis)[:16],
        variable_id=str(variable_id),
        complexity_tier=tier.value,
        complexity_vector=complexity_vector,
        inputs=tuple(inputs),
        parent_variables=tuple(parent_variables),
        action_inputs=tuple(action_inputs),
        latent_noise_inputs=tuple(latent_noise_inputs),
        terms=tuple(terms),
        transforms=tuple(transforms),
        temporal_lags=tuple(temporal_lags),
        sympy_expression=str(expression),
        metrics=metrics,
    )
    return spec.as_payload()


def validate_structural_equation_spec(payload: dict[str, Any]) -> dict[str, bool]:
    metrics = dict(payload.get("metrics") or {})
    tier = str(payload.get("complexity_tier") or "")
    return {
        "schema_version": payload.get("schema_version")
        == STRUCTURAL_EQUATION_SCHEMA_VERSION,
        "has_equation_id": bool(str(payload.get("equation_id") or "")),
        "has_variable_id": bool(str(payload.get("variable_id") or "")),
        "known_tier": tier in {item.value for item in ScmComplexityTier},
        "has_expression": bool(str(payload.get("sympy_expression") or "")),
        "has_metrics": bool(metrics),
        "has_complexity_vector": bool(payload.get("complexity_vector")),
        "metrics_have_degree": "polynomial_degree" in metrics,
        "metrics_have_transform_count": "nonlinear_transform_count" in metrics,
    }


def evaluate_structural_equation(
    payload: dict[str, Any], values: dict[str, float]
) -> float:
    inputs = [str(item) for item in payload.get("inputs") or []]
    local_symbols = {name: sp.Symbol(name) for name in inputs}
    expression = sp.sympify(
        str(payload.get("sympy_expression") or "0"), locals=local_symbols
    )
    fn = sp.lambdify(
        [local_symbols[name] for name in inputs], expression, modules="numpy"
    )
    result = fn(*[float(values.get(name, 0.0)) for name in inputs])
    return float(np.asarray(result).reshape(-1)[0])


def graph_complexity_metrics(
    graph: CausalGraphSpec, *, complexity_tier: str
) -> dict[str, Any]:
    nx_graph = nx.DiGraph()
    nx_graph.add_nodes_from(node.node_id for node in graph.nodes)
    nx_graph.add_edges_from(edge.as_pair() for edge in graph.edges)
    latent_nodes = {node.node_id for node in graph.nodes if not node.observed}
    observed_nodes = {node.node_id for node in graph.nodes if node.observed}
    latent_fanouts = [
        nx_graph.out_degree(node_id)
        for node_id in latent_nodes
        if nx_graph.out_degree(node_id) > 1
    ]
    return {
        "schema_version": SCM_COMPLEXITY_SCHEMA_VERSION,
        "complexity_tier": _coerce_tier(complexity_tier).value,
        "node_count": int(nx_graph.number_of_nodes()),
        "edge_count": int(nx_graph.number_of_edges()),
        "is_dag": bool(nx.is_directed_acyclic_graph(nx_graph)),
        "longest_path_length": _longest_path_length(nx_graph),
        "collider_count": _collider_count(nx_graph),
        "latent_confounder_count": len(latent_fanouts),
        "max_latent_fanout": int(max(latent_fanouts) if latent_fanouts else 0),
        "observed_node_count": len(observed_nodes),
        "latent_node_count": len(latent_nodes),
    }


def structural_equation_metrics(
    *,
    expression: sp.Expr,
    inputs: tuple[str, ...],
    latent_noise_inputs: tuple[str, ...],
    temporal_lags: tuple[TemporalLagSpec, ...],
    transforms: tuple[NonlinearTransformSpec, ...],
    terms: tuple[StructuralTermSpec, ...] = (),
) -> StructuralEquationMetrics:
    input_symbols = [sp.Symbol(name) for name in inputs]
    term_degree = max(
        (sum(int(power) for power in term.powers) for term in terms),
        default=0,
    )
    try:
        polynomial = sp.Poly(expression, *input_symbols)
        degree = int(polynomial.total_degree())
    except (PolynomialError, GeneratorsNeeded):
        degree = int(term_degree)
    interaction_term_count = sum(
        1
        for term in terms
        if term.term_type in {"interaction", "composition"} or len(term.variables) >= 2
    )
    return StructuralEquationMetrics(
        polynomial_degree=degree,
        input_count=len(inputs),
        interaction_term_count=int(interaction_term_count),
        nonlinear_transform_count=len(transforms),
        threshold_count=int(
            sum(1 for item in transforms if item.transform == "threshold")
        ),
        latent_input_count=len(latent_noise_inputs),
        temporal_lag_depth=int(
            max((int(item.lag) for item in temporal_lags), default=0)
        ),
    )


def _build_expression(
    *,
    variable_id: str,
    inputs: list[str],
    latent_noise_inputs: tuple[str, ...],
    tier: ScmComplexityTier,
    cfg: StructuralEquationConfig,
    complexity_vector: ScmComplexityVector,
) -> tuple[
    sp.Expr,
    list[StructuralTermSpec],
    list[NonlinearTransformSpec],
    list[TemporalLagSpec],
]:
    del tier  # Retained for source-compatible internal call signatures.
    vector = complexity_vector.as_payload()
    symbols = {name: sp.Symbol(name) for name in inputs}
    terms: list[StructuralTermSpec] = []
    transforms: list[NonlinearTransformSpec] = []
    temporal_lags: list[TemporalLagSpec] = []
    expression: sp.Expr = sp.Float(0.0)
    for index, name in enumerate(inputs):
        coefficient = round(1.0 / float(index + 1), 6)
        expression += coefficient * symbols[name]
        terms.append(
            StructuralTermSpec(
                term_type="linear",
                coefficient=coefficient,
                variables=(name,),
                powers=(1,),
            )
        )

    interaction_order = max(
        int(round(vector["polynomial_order"])), int(cfg.interaction_order), 1
    )
    pair_budget = int(round(vector["interaction_strength"] * max(1, len(inputs) - 1)))
    if vector["interaction_strength"] > 0.0 and len(inputs) >= 2:
        pair_count = min(len(inputs) - 1, max(1, pair_budget, interaction_order - 1))
        for index in range(pair_count):
            left = inputs[index]
            right = inputs[index + 1]
            coefficient = round(0.25 / float(index + 1), 6)
            expression += coefficient * symbols[left] * symbols[right]
            terms.append(
                StructuralTermSpec(
                    term_type="interaction",
                    coefficient=coefficient,
                    variables=(left, right),
                    powers=(1, 1),
                )
            )

    transform_count = max(
        int(cfg.nonlinear_transform_count),
        int(round(vector["nonlinear_strength"] * max(1, len(inputs)))),
    )
    if vector["nonlinear_strength"] > 0.0 and inputs:
        selected = inputs[: min(len(inputs), max(1, transform_count))]
        for index, name in enumerate(selected):
            transform = ("sin", "exp", "threshold")[index % 3]
            if transform == "sin":
                expression += sp.Float(0.15) * sp.sin(symbols[name])
            elif transform == "exp":
                expression += sp.Float(0.05) * sp.exp(sp.Float(0.1) * symbols[name])
            else:
                expression += sp.Float(0.2) / (1 + sp.exp(-symbols[name]))
            transforms.append(
                NonlinearTransformSpec(transform=transform, variable=name)
            )

    if vector["latent_confounding_strength"] > 0.0:
        for latent in latent_noise_inputs:
            if latent in symbols:
                coefficient = round(0.3 * vector["latent_confounding_strength"], 6)
                expression += coefficient * symbols[latent]
                terms.append(
                    StructuralTermSpec(
                        term_type="latent_confounder",
                        coefficient=coefficient,
                        variables=(latent,),
                        powers=(1,),
                    )
                )

    temporal_lag = max(
        int(cfg.temporal_lag), int(round(vector["temporal_memory_strength"] * 3.0))
    )
    if vector["temporal_memory_strength"] > 0.0 and temporal_lag > 0:
        lag_name = _lag_variable(variable_id, temporal_lag)
        lag_symbol = sp.Symbol(lag_name)
        coefficient = round(0.4 * vector["temporal_memory_strength"], 6)
        expression += coefficient * lag_symbol
        inputs.append(lag_name)
        temporal_lags.append(
            TemporalLagSpec(
                variable=str(variable_id), lag=int(temporal_lag), input=lag_name
            )
        )
        terms.append(
            StructuralTermSpec(
                term_type="temporal_lag",
                coefficient=coefficient,
                variables=(lag_name,),
                powers=(1,),
            )
        )

    if vector["hierarchy_depth"] > 0.0 and len(inputs) >= 2:
        module_name = f"{inputs[0]}__module"
        coefficient = round(0.1 * vector["hierarchy_depth"], 6)
        expression += coefficient * (sp.Symbol(inputs[0]) + sp.Symbol(inputs[-1])) ** 2
        transforms.append(
            NonlinearTransformSpec(
                transform="compositional_square_module", variable=module_name
            )
        )
        terms.append(
            StructuralTermSpec(
                term_type="composition",
                coefficient=coefficient,
                variables=(inputs[0], inputs[-1]),
                powers=(2,),
            )
        )

    return expression, terms, transforms, temporal_lags


def _interaction_term_count(expression: sp.Expr, input_symbols: list[sp.Symbol]) -> int:
    if not input_symbols:
        return 0
    try:
        polynomial = sp.Poly(expression, *input_symbols)
    except sp.PolynomialError:
        return 0
    count = 0
    for monomial, coefficient in zip(
        polynomial.monoms(), polynomial.coeffs(), strict=True
    ):
        if coefficient == 0:
            continue
        if sum(1 for power in monomial if power > 0) >= 2:
            count += 1
    return count


def _longest_path_length(graph: nx.DiGraph) -> int:
    if not nx.is_directed_acyclic_graph(graph) or graph.number_of_nodes() == 0:
        return 0
    return int(nx.dag_longest_path_length(graph))


def _collider_count(graph: nx.DiGraph) -> int:
    return int(sum(1 for node in graph.nodes if graph.in_degree(node) >= 2))


def _coerce_tier(value: str) -> ScmComplexityTier:
    try:
        return ScmComplexityTier(str(value))
    except ValueError:
        return ScmComplexityTier.TIER_0_LINEAR


def _ordered_unique(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value)
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _lag_variable(variable_id: str, lag: int) -> str:
    return f"{variable_id}_lag_{max(1, int(lag))}"


def _count_to_strength(value: int) -> float:
    return clamp01(float(value) / 3.0)


def _clamp_nonnegative(value: float) -> float:
    return max(0.0, float(value))
