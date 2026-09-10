"""Finite, certified full-history versus quotient-cache control."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, cast

SCHEMA_VERSION = "causal-observation-reproduction.cache-control.v1"
Action = Literal["acquire", "abstain"]
Regime = Literal["valid", "invalid"]
TiePolicy = Literal["acquire", "abstain"]


@dataclass(frozen=True)
class CacheControlConfig:
    """Outcome-blind finite candidate-set configuration."""

    nuisance_cardinalities: tuple[int, ...] = (1, 4, 16)
    acquisition_cost: float = 0.25
    tie_policy: TiePolicy = "abstain"

    def __post_init__(self) -> None:
        if not self.nuisance_cardinalities or any(
            cardinality < 1 for cardinality in self.nuisance_cardinalities
        ):
            msg = "nuisance cardinalities must be non-empty positive integers"
            raise ValueError(msg)
        if len(set(self.nuisance_cardinalities)) != len(self.nuisance_cardinalities):
            msg = "nuisance cardinalities must not contain duplicates"
            raise ValueError(msg)
        if self.acquisition_cost < 0.0:
            msg = "acquisition cost must be non-negative"
            raise ValueError(msg)
        if self.tie_policy not in {"acquire", "abstain"}:
            msg = "tie policy must be acquire or abstain"
            raise ValueError(msg)


@dataclass(frozen=True)
class CandidateHistory:
    signal: int
    nuisance: int

    def full_key(self) -> tuple[int, int]:
        return self.signal, self.nuisance

    def quotient_key(self) -> tuple[int]:
        return (self.signal,)


def run_cache_control(config: CacheControlConfig | None = None) -> dict[str, Any]:
    """Compare cache keys under one exact evaluator and explicit falsifier."""

    resolved = config or CacheControlConfig()
    rows: list[dict[str, object]] = []
    for cardinality in resolved.nuisance_cardinalities:
        histories = tuple(
            CandidateHistory(signal=signal, nuisance=nuisance)
            for signal in (0, 1)
            for nuisance in range(cardinality)
        )
        regimes: tuple[Regime, ...] = (
            ("valid",) if cardinality == 1 else ("valid", "invalid")
        )
        rows.extend(_comparison_row(histories, regime, resolved) for regime in regimes)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "config": asdict(resolved),
        "evaluation_domain": "finite planner candidate set, not a single online policy history",
        "shared_evaluator": "exact action-value evaluator",
        "cache_reuse_policy": "both arms cache; only the cache key differs",
        "dataset_layout": {
            "manifest": "manifest.json",
            "rows": "rows.jsonl",
            "summary": "summary.json",
        },
    }
    manifest["config_digest"] = sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"manifest": manifest, "rows": rows, "summary": _summary(rows)}


def write_cache_control_dataset(output_dir: str | Path) -> dict[str, Path]:
    """Write an inspection-friendly manifest, flat JSONL table, and summary."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    result = run_cache_control()
    paths = {
        name: directory / name
        for name in ("manifest.json", "rows.jsonl", "summary.json")
    }
    paths["manifest.json"].write_text(
        json.dumps(result["manifest"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["rows.jsonl"].write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in result["rows"]),
        encoding="utf-8",
    )
    paths["summary.json"].write_text(
        json.dumps(result["summary"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return paths


def _comparison_row(
    histories: tuple[CandidateHistory, ...], regime: Regime, config: CacheControlConfig
) -> dict[str, object]:
    certified = _certify(histories, regime, config.acquisition_cost)
    full, full_calls = _evaluate(histories, regime, config, mode="full")
    quotient, quotient_calls = _evaluate(histories, regime, config, mode="quotient")
    disagreements = sum(
        left != right for left, right in zip(full, quotient, strict=True)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "regime": regime,
        "nuisance_cardinality": len(histories) // 2,
        "candidate_history_count": len(histories),
        "projection_certified": certified,
        "decision_preserved": disagreements == 0,
        "action_value_or_decision_disagreements": disagreements,
        "full_history_unique_evaluator_calls": full_calls,
        "quotient_cache_unique_evaluator_calls": quotient_calls,
        "evidence_eligibility": "positive_compression_control"
        if certified
        else "invalid_projection_control",
    }


def _values(
    history: CandidateHistory, regime: Regime, cost: float
) -> tuple[float, float]:
    acquire = 1.0 - cost if history.signal else -cost
    if regime == "invalid" and history.nuisance % 2:
        acquire = -acquire
    return acquire, 0.0


def _certify(
    histories: tuple[CandidateHistory, ...], regime: Regime, cost: float
) -> bool:
    representatives: dict[tuple[int], tuple[float, float]] = {}
    for history in histories:
        values = _values(history, regime, cost)
        representatives.setdefault(history.quotient_key(), values)
        if representatives[history.quotient_key()] != values:
            return False
    return True


def _evaluate(
    histories: tuple[CandidateHistory, ...],
    regime: Regime,
    config: CacheControlConfig,
    *,
    mode: Literal["full", "quotient"],
) -> tuple[list[tuple[float, float, Action]], int]:
    cache: dict[tuple[int, ...], tuple[float, float]] = {}
    records: list[tuple[float, float, Action]] = []
    evaluator_calls = 0
    for history in histories:
        key = history.full_key() if mode == "full" else history.quotient_key()
        values = cache.get(key)
        if values is None:
            values = _values(history, regime, config.acquisition_cost)
            cache[key] = values
            evaluator_calls += 2
        records.append((*values, _choose_action(values, config.tie_policy)))
    return records, evaluator_calls


def _choose_action(values: tuple[float, float], tie_policy: TiePolicy) -> Action:
    """Choose deterministically, including the contract's declared tie policy."""

    if values[0] == values[1]:
        return tie_policy
    return "acquire" if values[0] > values[1] else "abstain"


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    valid = [row for row in rows if row["regime"] == "valid"]
    invalid = [row for row in rows if row["regime"] == "invalid"]
    largest = max(valid, key=lambda row: cast("int", row["nuisance_cardinality"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "largest_valid_cardinality": largest["nuisance_cardinality"],
        "full_history_unique_evaluator_calls": largest[
            "full_history_unique_evaluator_calls"
        ],
        "quotient_cache_unique_evaluator_calls": largest[
            "quotient_cache_unique_evaluator_calls"
        ],
        "all_valid_rows_certified_and_preserved": all(
            bool(row["projection_certified"]) and bool(row["decision_preserved"])
            for row in valid
        ),
        "all_invalid_rows_falsify_projection": all(
            not bool(row["projection_certified"])
            and not bool(row["decision_preserved"])
            for row in invalid
        ),
    }
