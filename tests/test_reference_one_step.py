"""Exact one-step checks, enabled when the learned extra is installed."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

pytest.importorskip("pandas")

from causal_observation_reproduction.reference.learned_coo_gate import (
    LearnedCOODatasetBuilder,
    LearnedCOOGateConfig,
    build_grouped_splits,
)

PROJECT_ROOT = Path(__file__).parents[1]


def test_checked_one_step_config_equals_source_derived_defaults() -> None:
    checked = json.loads(
        (PROJECT_ROOT / "configs/reference/one_step_gate.json").read_text(
            encoding="utf-8"
        )
    )
    normalized = json.loads(json.dumps(asdict(LearnedCOOGateConfig()), sort_keys=True))
    assert checked == normalized


def test_one_step_materialization_preserves_exact_source_cardinality() -> None:
    config = LearnedCOOGateConfig()
    frame, manifest = LearnedCOODatasetBuilder(config).materialize()
    splits = build_grouped_splits(frame, config=config)

    assert len(frame) == manifest["row_count"] == 274
    assert frame["experiment_panel"].value_counts().to_dict() == {
        "development": 180,
        "frozen_test": 60,
        "primary": 20,
        "confirmation": 14,
    }
    assert {name: len(values) for name, values in splits.items()} == {
        "train": 144,
        "validation": 36,
        "frozen_test": 60,
    }
    assert manifest["source_revision"] == ("b4c468ef4dfac5eeab8b11a9e7f990ba36824748")
