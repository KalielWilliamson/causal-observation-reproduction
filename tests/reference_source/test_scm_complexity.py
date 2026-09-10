from __future__ import annotations

import numpy as np

from causal_observation_reproduction.reference.artifacts import assert_json_roundtrip
from causal_observation_reproduction.reference.scm_complexity import (
    ScmComplexityTier,
    ScmComplexityVector,
    StructuralEquationConfig,
    StructuralEquationMetrics,
    StructuralEquationSpec,
    StructuralTermSpec,
    TemporalLagSpec,
    build_structural_equation_spec,
    complexity_vector_from_matrix,
    evaluate_structural_equation,
    validate_structural_equation_spec,
)


def test_structural_equation_ladder_adds_interactions_and_nonlinearity() -> None:
    linear = build_structural_equation_spec(
        variable_id="y",
        parent_variables=("x1", "x2", "x3"),
        config=StructuralEquationConfig(
            complexity_tier=ScmComplexityTier.TIER_0_LINEAR.value
        ),
    )
    nonlinear = build_structural_equation_spec(
        variable_id="y",
        parent_variables=("x1", "x2", "x3"),
        config=StructuralEquationConfig(
            complexity_tier=ScmComplexityTier.TIER_2_SMOOTH_NONLINEAR.value,
            interaction_order=3,
            nonlinear_transform_count=3,
        ),
    )

    assert all(validate_structural_equation_spec(linear).values())
    assert all(validate_structural_equation_spec(nonlinear).values())
    assert linear["metrics"]["interaction_term_count"] == 0
    assert nonlinear["metrics"]["interaction_term_count"] >= 1
    assert nonlinear["metrics"]["nonlinear_transform_count"] == 3
    assert (
        evaluate_structural_equation(nonlinear, {"x1": 1.0, "x2": 2.0, "x3": -0.5})
        != 0.0
    )


def test_temporal_tier_declares_lagged_inputs() -> None:
    equation = build_structural_equation_spec(
        variable_id="state",
        parent_variables=("parent",),
        config=StructuralEquationConfig(
            complexity_tier=ScmComplexityTier.TIER_4_TEMPORAL.value,
            temporal_lag=2,
        ),
    )

    assert all(validate_structural_equation_spec(equation).values())
    assert equation["metrics"]["temporal_lag_depth"] == 2
    assert equation["temporal_lags"] == [
        {"variable": "state", "lag": 2, "input": "state_lag_2"}
    ]
    assert "state_lag_2" in equation["inputs"]


def test_numpy_design_matrix_drives_complexity_vector() -> None:
    matrix = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0],
            [3.0, 1.0, 1.0, 0.5, 0.8, 0.25, 0.20, 0.0],
        ]
    )
    vector = complexity_vector_from_matrix(matrix, row_index=1)
    equation = build_structural_equation_spec(
        variable_id="state",
        parent_variables=("x1", "x2", "x3"),
        config=StructuralEquationConfig(complexity_vector=vector),
    )

    assert equation["complexity_vector"]["interaction_strength"] == 1.0
    assert equation["metrics"]["interaction_term_count"] >= 1
    assert equation["metrics"]["nonlinear_transform_count"] >= 1
    assert equation["metrics"]["temporal_lag_depth"] >= 1


def test_typed_structural_equation_spec_serializes_to_existing_payload_contract() -> (
    None
):
    vector = ScmComplexityVector(
        polynomial_order=2.0, interaction_strength=1.0, temporal_memory_strength=0.5
    )
    spec = StructuralEquationSpec(
        equation_id="generated-scm-equation-test",
        variable_id="state",
        complexity_tier=ScmComplexityTier.TIER_4_TEMPORAL.value,
        complexity_vector=vector,
        inputs=("parent", "state_lag_2"),
        parent_variables=("parent",),
        action_inputs=(),
        latent_noise_inputs=(),
        terms=(
            StructuralTermSpec(
                term_type="linear", coefficient=1.0, variables=("parent",), powers=(1,)
            ),
            StructuralTermSpec(
                term_type="temporal_lag",
                coefficient=0.2,
                variables=("state_lag_2",),
                powers=(1,),
            ),
        ),
        transforms=(),
        temporal_lags=(TemporalLagSpec(variable="state", lag=2, input="state_lag_2"),),
        sympy_expression="1.0*parent + 0.2*state_lag_2",
        metrics=StructuralEquationMetrics(
            polynomial_degree=1,
            input_count=2,
            interaction_term_count=0,
            nonlinear_transform_count=0,
            threshold_count=0,
            latent_input_count=0,
            temporal_lag_depth=2,
        ),
    )

    payload = spec.as_payload()

    assert assert_json_roundtrip(payload) == payload
    assert all(validate_structural_equation_spec(payload).values())
    assert payload["terms"][1] == {
        "term_type": "temporal_lag",
        "coefficient": 0.2,
        "variables": ["state_lag_2"],
        "powers": [1],
    }
    assert payload["temporal_lags"] == [
        {"variable": "state", "lag": 2, "input": "state_lag_2"}
    ]
    assert (
        evaluate_structural_equation(payload, {"parent": 1.0, "state_lag_2": 2.0})
        == 1.4
    )
