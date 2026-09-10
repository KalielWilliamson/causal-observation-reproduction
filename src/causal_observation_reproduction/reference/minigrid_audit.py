"""Read and export the exact 16-row MiniGrid appendix audit."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from enum import StrEnum
from importlib.resources import files
from pathlib import Path
from statistics import fmean
from typing import Any

SOURCE_REVISION = "b4c468ef4dfac5eeab8b11a9e7f990ba36824748"


class MiniGridRelation(StrEnum):
    """The four source--target relations reported in the appendix."""

    IDENTITY = "same_env_same_seed"
    SEED_SHIFT = "same_env_seed_shift"
    SCALE_CHANGE = "same_family_scale_change"
    NEGATIVE_CONTROL = "cross_family_causal_negative_control"


@dataclass(frozen=True)
class MiniGridAuditRow:
    """Paper-facing fields from one evaluated source--target policy row."""

    pair_id: str
    policy_acquisition_mode: str
    relation: MiniGridRelation
    source_env_id: str
    target_env_id: str
    source_seed: int
    target_seed: int
    readiness: float
    transferred: bool


@dataclass(frozen=True)
class MiniGridRelationSummary:
    """Descriptive appendix aggregate for one declared relation."""

    relation: MiniGridRelation
    row_count: int
    transferred: int
    mean_readiness: float


def load_minigrid_external_control() -> tuple[MiniGridAuditRow, ...]:
    """Load the checked, source-derived evaluated rows without optional packages."""

    resource = files("causal_observation_reproduction.reference").joinpath(
        "data/minigrid_external_control.jsonl"
    )
    rows: list[MiniGridAuditRow] = []
    for line in resource.read_text(encoding="utf-8").splitlines():
        payload: dict[str, Any] = json.loads(line)
        rows.append(
            MiniGridAuditRow(
                pair_id=str(payload["pair_id"]),
                policy_acquisition_mode=str(payload["policy_acquisition_mode"]),
                relation=MiniGridRelation(str(payload["perturbation_type"])),
                source_env_id=str(payload["source_env_id"]),
                target_env_id=str(payload["target_env_id"]),
                source_seed=int(payload["source_seed"]),
                target_seed=int(payload["target_seed"]),
                readiness=float(payload["transfer_readiness_score"]),
                transferred=bool(payload["observed_transfer_success"]),
            )
        )
    if len(rows) != 16:
        raise ValueError(f"expected 16 evaluated MiniGrid rows, found {len(rows)}")
    return tuple(rows)


def summarize_minigrid_external_control(
    rows: tuple[MiniGridAuditRow, ...] | None = None,
) -> tuple[MiniGridRelationSummary, ...]:
    """Recompute the four appendix aggregates from the evaluated rows."""

    grouped: dict[MiniGridRelation, list[MiniGridAuditRow]] = defaultdict(list)
    for row in rows or load_minigrid_external_control():
        grouped[row.relation].append(row)
    return tuple(
        MiniGridRelationSummary(
            relation=relation,
            row_count=len(group),
            transferred=sum(row.transferred for row in group),
            mean_readiness=fmean(row.readiness for row in group),
        )
        for relation, group in sorted(grouped.items(), key=lambda item: item[0].value)
    )


def write_minigrid_external_control(output_dir: str | Path) -> tuple[Path, ...]:
    """Export the inspectable rows, analysis-ready Parquet, and derived summary."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    package = files("causal_observation_reproduction.reference").joinpath("data")
    jsonl_path = destination / "minigrid_external_control.jsonl"
    parquet_path = destination / "minigrid_external_control.parquet"
    summary_path = destination / "minigrid_external_control_summary.json"
    jsonl_path.write_text(
        package.joinpath("minigrid_external_control.jsonl").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    parquet_path.write_bytes(
        package.joinpath("minigrid_external_control.parquet").read_bytes()
    )
    summary = {
        "schema_version": "causal-observation-reproduction.minigrid-external-control.v1",
        "source_revision": SOURCE_REVISION,
        "row_count": 16,
        "relations": [
            {**asdict(row), "relation": row.relation.value}
            for row in summarize_minigrid_external_control()
        ],
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return jsonl_path, parquet_path, summary_path


__all__ = [
    "MiniGridAuditRow",
    "MiniGridRelation",
    "MiniGridRelationSummary",
    "load_minigrid_external_control",
    "summarize_minigrid_external_control",
    "write_minigrid_external_control",
]
