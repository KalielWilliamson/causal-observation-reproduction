import json

import pytest

from causal_observation_reproduction.certified_cache_control import (
    CacheControlConfig,
    run_cache_control,
    write_cache_control_dataset,
)


def test_valid_control_preserves_decisions_and_compresses_exact_evaluations() -> None:
    result = run_cache_control(CacheControlConfig(nuisance_cardinalities=(4,)))
    valid = next(row for row in result["rows"] if row["regime"] == "valid")
    assert valid["projection_certified"] is True
    assert valid["decision_preserved"] is True
    assert valid["full_history_unique_evaluator_calls"] == 16
    assert valid["quotient_cache_unique_evaluator_calls"] == 4


def test_invalid_control_falsifies_the_same_projection() -> None:
    result = run_cache_control(CacheControlConfig(nuisance_cardinalities=(4,)))
    invalid = next(row for row in result["rows"] if row["regime"] == "invalid")
    assert invalid["projection_certified"] is False
    assert invalid["decision_preserved"] is False
    assert invalid["action_value_or_decision_disagreements"] > 0


def test_dataset_is_manifest_plus_jsonl_rows_plus_summary(tmp_path) -> None:
    paths = write_cache_control_dataset(tmp_path)
    assert set(paths) == {"manifest.json", "rows.jsonl", "summary.json"}
    manifest = json.loads(paths["manifest.json"].read_text(encoding="utf-8"))
    summary = json.loads(paths["summary.json"].read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in paths["rows.jsonl"].read_text(encoding="utf-8").splitlines()
    ]
    assert manifest["dataset_layout"]["rows"] == "rows.jsonl"
    assert summary["all_valid_rows_certified_and_preserved"] is True
    assert len(rows) == 5


@pytest.mark.parametrize(
    ("nuisance_cardinalities", "match"),
    [
        ((), "non-empty"),
        ((0,), "positive"),
        ((4, 4), "duplicates"),
    ],
)
def test_cache_control_rejects_ambiguous_evaluation_domains(
    nuisance_cardinalities: tuple[int, ...], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        CacheControlConfig(nuisance_cardinalities=nuisance_cardinalities)
