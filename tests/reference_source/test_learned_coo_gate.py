from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pandas")

from causal_observation_reproduction.reference.learned_coo_gate import (
    LABEL_DENYLIST,
    MODEL_FEATURE_ALLOWLIST,
    LearnedCOODatasetBuilder,
    LearnedCOOGateConfig,
    assert_split_overlap_free,
    build_grouped_splits,
    model_feature_matrix,
    policy_metrics,
)


def _config() -> LearnedCOOGateConfig:
    return LearnedCOOGateConfig(
        development_seeds=(11, 12, 13, 14),
        frozen_test_seeds=(21, 22),
        historical_primary_seeds=(1,),
        historical_confirmation_seeds=(101,),
        null_primary_seeds=(1,),
        null_confirmation_seeds=(201,),
        pathology_primary_seeds=(1,),
        bootstrap_samples=8,
    )


def test_canonical_net_value_subtracts_cost_once_and_raw_semantics_are_explicit() -> (
    None
):
    frame, _ = LearnedCOODatasetBuilder(_config()).materialize()
    assert np.allclose(
        frame["net_refinement_value"],
        frame["gross_refinement_value"]
        - frame["observation_cost"]
        - frame["probe_cost"],
    )
    assert frame.loc[
        frame["registered_region"].eq("failure"),
        "observation_cost_included_in_raw_metric",
    ].all()
    assert not frame.loc[
        frame["registered_region"].ne("failure"),
        "observation_cost_included_in_raw_metric",
    ].any()


def test_oracle_threshold_eiv_regret_and_error_losses() -> None:
    values = np.asarray([2.0, -3.0, 0.0])
    predicted = np.asarray([1.0, 1.0, -1.0])
    metrics = policy_metrics(values, predicted)
    assert metrics["eiv"] == -1.0 / 3.0
    assert metrics["mean_regret"] == 1.0
    assert metrics["false_positive_loss"] == 1.0
    assert metrics["false_negative_loss"] == 0.0
    oracle = policy_metrics(values, values)
    assert oracle["mean_regret"] == 0.0
    assert oracle["activation_rate"] == 1.0 / 3.0


def test_features_are_allowlisted_and_observable_only() -> None:
    frame, _ = LearnedCOODatasetBuilder(_config()).materialize()
    assert not (set(MODEL_FEATURE_ALLOWLIST) & set(LABEL_DENYLIST))
    matrix = model_feature_matrix(frame)
    assert matrix.shape == (len(frame), len(MODEL_FEATURE_ALLOWLIST))
    assert all("latent" not in graph.lower() for graph in frame["graph_json"])


def test_grouped_splits_are_overlap_free_and_deterministic() -> None:
    frame, _ = LearnedCOODatasetBuilder(_config()).materialize()
    first = build_grouped_splits(frame, config=_config())
    second = build_grouped_splits(frame, config=_config())
    assert first == second
    assert_split_overlap_free(frame, first)
