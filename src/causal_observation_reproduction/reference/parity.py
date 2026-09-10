"""Golden checks between source-derived executions and manuscript data."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from causal_observation_reproduction.reference.sequential_experiment import (
    ConditionalEfficiencyConfirmationResult,
)


@dataclass(frozen=True)
class SequentialParityReport:
    """Outcome of the locked source-to-public comparison."""

    source_revision: str
    comparison_count: int
    paired_evaluations_per_budget_regime: int
    topology_clusters: int
    primary_noninferior: bool
    information_action_agreement_count: int
    passed: bool


@dataclass(frozen=True)
class OneStepParityReport:
    """Outcome and known cardinality discrepancy for the one-step pilot."""

    source_revision: str
    executable_total_rows: int
    development_rows: int
    historical_rows: int
    frozen_test_rows: int
    manuscript_claimed_learning_rows: int
    manuscript_claimed_total_rows: int
    metric_conclusions_match: bool
    corpus_cardinality_matches: bool
    assessment: str


def verify_online_compute_counts(payload: dict[str, Any]) -> None:
    """Check published work counts while allowing explicit zero-valued fields.

    The pinned implementation emits two non-applicable generic-arm counters as
    zeros, while the manuscript summary predates those schema fields and omits
    them.  Projecting onto the published fields preserves exact scientific
    parity; any unknown non-zero counter still fails closed.
    """

    expected = _reference_payload("online_compute_reference.json")[
        "counted_online_work"
    ]
    observed = payload.get("counted_online_work")
    if not isinstance(observed, Mapping) or observed.keys() != expected.keys():
        raise ValueError("online compute counts differ from the manuscript reference")
    for arm, expected_counts in expected.items():
        observed_counts = observed.get(arm)
        if not isinstance(expected_counts, Mapping) or not isinstance(
            observed_counts, Mapping
        ):
            raise ValueError(
                "online compute counts differ from the manuscript reference"
            )
        if {
            field: observed_counts.get(field) for field in expected_counts
        } != expected_counts:
            raise ValueError(
                "online compute counts differ from the manuscript reference"
            )
        if any(
            value not in (0, 0.0, None)
            for field, value in observed_counts.items()
            if field not in expected_counts
        ):
            raise ValueError(
                "online compute counts differ from the manuscript reference"
            )


def _reference_payload(name: str) -> dict[str, Any]:
    resource = files("causal_observation_reproduction.reference").joinpath(
        f"data/{name}"
    )
    payload: dict[str, Any] = json.loads(resource.read_text(encoding="utf-8"))
    return payload


def verify_locked_sequential_confirmation(
    result: ConditionalEfficiencyConfirmationResult,
    *,
    output_path: str | Path | None = None,
) -> SequentialParityReport:
    """Fail unless a full locked run agrees with every published comparison."""

    expected = _reference_payload("sequential_confirmation_reference.json")
    actual: dict[str, Any] = json.loads(
        Path(result.artifacts["paper_projection"]).read_text(encoding="utf-8")
    )
    expected_by_key = {
        (str(row["regime"]), int(row["budget"])): row for row in expected["comparisons"]
    }
    actual_by_key = {
        (str(row["regime"]), int(row["budget"])): row for row in actual["comparisons"]
    }
    if actual_by_key.keys() != expected_by_key.keys():
        raise ValueError(
            "sequential comparison cells differ from the manuscript reference"
        )
    float_fields = (
        "return_gap_quotient_minus_generic",
        "quotient_mean_rollouts",
        "generic_mean_rollouts",
        "quotient_mean_branches",
        "generic_mean_branches",
    )
    integer_fields = ("quotient_better", "equal", "quotient_worse")
    for key, reference in expected_by_key.items():
        observed = actual_by_key[key]
        for field in float_fields:
            if not math.isclose(
                float(observed[field]),
                float(reference[field]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"sequential parity mismatch for {key!r} field {field!r}"
                )
        for field in integer_fields:
            if int(observed[field]) != int(reference[field]):
                raise ValueError(
                    f"sequential parity mismatch for {key!r} field {field!r}"
                )
        expected_pairs = int(expected["scope"]["paired_evaluations_per_budget_regime"])
        if int(observed["paired_evaluation_count"]) != expected_pairs:
            raise ValueError(f"sequential pair count mismatch for {key!r}")

    inference: list[dict[str, Any]] = json.loads(
        Path(result.artifacts["clustered_noninferiority"]).read_text(encoding="utf-8")
    )
    primary_budget = int(expected["primary_noninferiority"]["budget"])
    primary = next(
        row
        for row in inference
        if row["comparator"] == "generic_bounded_rollout_voi"
        and int(row["generic_rollout_budget"]) == primary_budget
    )
    primary_expected = expected["primary_noninferiority"]
    if bool(primary["noninferior"]) is not bool(primary_expected["noninferior"]):
        raise ValueError("primary noninferiority decision differs from the manuscript")
    for actual_key, expected_key in (
        ("mean_paired_difference", "quotient_minus_generic_mean"),
        ("bootstrap_ci_low", "bootstrap_ci"),
        ("bootstrap_ci_high", "bootstrap_ci"),
    ):
        expected_value = (
            primary_expected[expected_key][0 if actual_key.endswith("low") else 1]
            if expected_key == "bootstrap_ci"
            else primary_expected[expected_key]
        )
        if not math.isclose(
            float(primary[actual_key]),
            float(expected_value),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"primary noninferiority mismatch for {actual_key!r}")

    gate_rows = {
        _paired_row_key(row): row
        for row in result.rows
        if row["policy"] == "learned_causal_quotient"
        and int(row["generic_rollout_budget"]) == 0
    }
    comparator_rows = {
        _paired_row_key(row): row
        for row in result.rows
        if row["policy"] == "generic_bounded_rollout_voi"
        and int(row["generic_rollout_budget"]) == primary_budget
    }
    if gate_rows.keys() != comparator_rows.keys():
        raise ValueError("primary information-action rows are not exactly paired")
    action_agreements = sum(
        int(gate_rows[key]["probe_count"]) == int(comparator_rows[key]["probe_count"])
        for key in gate_rows
    )
    if action_agreements != 13_824:
        raise ValueError(
            "primary information-action agreement differs from the manuscript"
        )

    report = SequentialParityReport(
        source_revision=str(expected["source_revision"]),
        comparison_count=len(actual_by_key),
        paired_evaluations_per_budget_regime=int(
            expected["scope"]["paired_evaluations_per_budget_regime"]
        ),
        topology_clusters=int(expected["scope"]["topology_clusters"]),
        primary_noninferior=bool(primary["noninferior"]),
        information_action_agreement_count=action_agreements,
        passed=True,
    )
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(asdict(report), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def verify_one_step_calibration(
    *,
    dataset_manifest_path: str | Path,
    split_manifest_path: str | Path,
    policy_metrics_path: str | Path,
    output_path: str | Path | None = None,
) -> OneStepParityReport:
    """Verify the manuscript's one-step conclusions and expose its row-count gap."""

    expected_revision = str(
        _reference_payload("sequential_confirmation_reference.json")["source_revision"]
    )
    dataset = json.loads(Path(dataset_manifest_path).read_text(encoding="utf-8"))
    split_manifest = json.loads(Path(split_manifest_path).read_text(encoding="utf-8"))
    if dataset.get("source_revision") != expected_revision:
        raise ValueError("one-step source revision differs from the pinned reference")
    total_rows = int(dataset.get("row_count", -1))
    if total_rows != 274:
        raise ValueError("one-step executable row count differs from the source")

    splits = split_manifest.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("one-step split manifest lacks split rows")
    split_rows: dict[str, list[str]] = {}
    for name in ("train", "validation", "frozen_test"):
        values = splits.get(name)
        if not isinstance(values, list) or not all(
            isinstance(value, str) for value in values
        ):
            raise ValueError(f"one-step split {name!r} is malformed")
        split_rows[name] = values
    if {name: len(values) for name, values in split_rows.items()} != {
        "train": 144,
        "validation": 36,
        "frozen_test": 60,
    }:
        raise ValueError("one-step grouped split cardinality differs")
    frozen_motifs = Counter(value.split(":")[1] for value in split_rows["frozen_test"])
    if frozen_motifs != {
        "decision-relevant-refinement": 20,
        "no-value-refinement": 20,
        "pathology-failure-region": 20,
    }:
        raise ValueError("one-step frozen motif balance differs from the manuscript")

    metrics: dict[str, dict[str, str | None]] = {}
    with Path(policy_metrics_path).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            policy = row.get("policy")
            if not policy or policy in metrics:
                raise ValueError(
                    "one-step policy metrics contain a missing or duplicate arm"
                )
            metrics[policy] = dict(row)
    expected_policies = {
        "never_refine",
        "always_refine",
        "linear_value_model",
        "formula_coo_oracle_statistic",
        "oracle_coo",
        "mlp_value_model",
        "gnn_coo",
        "budget_matched_random",
    }
    if metrics.keys() != expected_policies:
        raise ValueError("one-step policy metrics are incomplete")

    def metric(policy: str, field: str) -> float:
        value = metrics[policy].get(field)
        if value is None:
            raise ValueError(f"one-step policy {policy!r} lacks metric {field!r}")
        return float(value)

    oracle_eiv = metric("oracle_coo", "eiv")
    for policy in ("linear_value_model", "mlp_value_model", "gnn_coo"):
        if not math.isclose(metric(policy, "eiv"), oracle_eiv, abs_tol=1e-12):
            raise ValueError(f"one-step value conclusion differs for {policy!r}")
        if not math.isclose(metric(policy, "mean_regret"), 0.0, abs_tol=1e-12):
            raise ValueError(f"one-step regret conclusion differs for {policy!r}")
        if not math.isclose(
            metric(policy, "activation_rate"), 2.0 / 3.0, abs_tol=1e-12
        ):
            raise ValueError(f"one-step null-tie behavior differs for {policy!r}")
    if not math.isclose(
        metric("oracle_coo", "activation_rate"), 1.0 / 3.0, abs_tol=1e-12
    ):
        raise ValueError("one-step oracle activation differs from the manuscript")
    if not (
        metric("linear_value_model", "mae") < metric("mlp_value_model", "mae")
        and metric("linear_value_model", "mae") < metric("gnn_coo", "mae")
    ):
        raise ValueError("one-step linear-error conclusion differs from the manuscript")

    development_rows = len(split_rows["train"]) + len(split_rows["validation"])
    frozen_rows = len(split_rows["frozen_test"])
    report = OneStepParityReport(
        source_revision=expected_revision,
        executable_total_rows=total_rows,
        development_rows=development_rows,
        historical_rows=total_rows - development_rows - frozen_rows,
        frozen_test_rows=frozen_rows,
        manuscript_claimed_learning_rows=274,
        manuscript_claimed_total_rows=334,
        metric_conclusions_match=True,
        corpus_cardinality_matches=False,
        assessment="outcomes_agree_but_corpus_cardinality_is_disproved",
    )
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(asdict(report), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def _paired_row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        row[field]
        for field in (
            "topology_instance_id",
            "family_id",
            "seed",
            "regime",
            "distractor_count",
            "horizon",
            "probe_cost",
        )
    )


__all__ = [
    "OneStepParityReport",
    "SequentialParityReport",
    "verify_locked_sequential_confirmation",
    "verify_one_step_calibration",
    "verify_online_compute_counts",
]
