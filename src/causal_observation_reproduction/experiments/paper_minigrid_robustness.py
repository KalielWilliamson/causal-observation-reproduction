"""Independent-seed robustness extension for the frozen MiniGrid panel."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any, NoReturn

from causal_observation_reproduction.experiments.paper_minigrid_policy import (
    PaperMiniGridCell,
    PaperMiniGridIndependentSeeds,
    PaperMiniGridPpoConfig,
    evaluate_paper_minigrid_cell_independent,
    paper_minigrid_cells,
)

SCHEMA_VERSION = "causal-observation-reproduction.minigrid-robustness.v1"
EVIDENCE_STATUS = "independent_seed_robustness_pending_review"


def _raise_invalid(message: str) -> NoReturn:
    raise ValueError(message)


@dataclass(frozen=True)
class PaperMiniGridRobustnessConfig:
    """Disjoint randomization panels for an independent robustness extension."""

    ppo: PaperMiniGridPpoConfig = field(default_factory=PaperMiniGridPpoConfig)
    optimizer_seeds: tuple[int, ...] = (
        101,
        211,
        307,
        401,
        503,
        601,
        701,
        809,
        907,
        1009,
    )
    validation_seeds: tuple[int, ...] = tuple(range(6_001, 6_009))
    source_evaluation_seeds: tuple[int, ...] = tuple(range(7_001, 7_033))
    target_evaluation_seeds: tuple[int, ...] = tuple(range(8_001, 8_033))
    random_action_seeds: tuple[int, ...] = tuple(range(9_001, 9_033))

    def __post_init__(self) -> None:
        panels = (
            self.optimizer_seeds,
            self.validation_seeds,
            self.source_evaluation_seeds,
            self.target_evaluation_seeds,
            self.random_action_seeds,
        )
        if any(not panel for panel in panels):
            _raise_invalid("every MiniGrid robustness seed panel must be nonempty")
        if any(len(set(panel)) != len(panel) for panel in panels):
            _raise_invalid("MiniGrid robustness seeds must be unique within a panel")
        occupied: set[int] = set()
        for panel in panels:
            if occupied.intersection(panel):
                _raise_invalid("MiniGrid robustness seed panels must be disjoint")
            occupied.update(panel)
        if len(self.target_evaluation_seeds) != len(self.random_action_seeds):
            _raise_invalid("target environment and random-action seeds must align")


@dataclass(frozen=True)
class PaperMiniGridRobustnessRun:
    """One independent PPO training run, the unit of replication."""

    cell_id: str
    optimizer_seed: int
    source_environment: str
    target_environment: str
    source_layout_seed: int
    target_layout_seed: int
    perturbation_type: str
    control_role: str
    expected_transfer: bool
    source_success_rate: float
    target_zero_shot_success_rate: float
    target_random_success_rate: float
    source_efficiency_score: float
    target_zero_shot_efficiency_score: float
    target_random_efficiency_score: float
    transfer_effect_score: float
    source_mastered: bool
    observed_transfer_success: bool
    agrees_with_reference: bool
    evaluation_episodes_per_arm: int
    evidence_status: str = EVIDENCE_STATUS


@dataclass(frozen=True)
class PaperMiniGridRobustnessSummary:
    """Across-training-seed summary for one predeclared cell."""

    cell_id: str
    expected_transfer: bool
    training_run_count: int
    agreement_count: int
    agreement_rate: float
    agreement_ci_lower: float
    agreement_ci_upper: float
    observed_transfer_count: int
    mean_source_success_rate: float
    mean_target_zero_shot_success_rate: float
    mean_target_random_success_rate: float
    mean_transfer_effect_score: float


def run_paper_minigrid_robustness(
    config: PaperMiniGridRobustnessConfig | None = None,
    cells: tuple[PaperMiniGridCell, ...] | None = None,
) -> tuple[
    tuple[PaperMiniGridRobustnessRun, ...],
    tuple[PaperMiniGridRobustnessSummary, ...],
]:
    """Run disjoint optimizer/validation/evaluation seeds for every cell."""

    resolved = config or PaperMiniGridRobustnessConfig()
    resolved_cells = cells or paper_minigrid_cells()
    if not resolved_cells:
        _raise_invalid("the MiniGrid robustness panel must contain at least one cell")
    runs: list[PaperMiniGridRobustnessRun] = []
    for cell in resolved_cells:
        for optimizer_seed in resolved.optimizer_seeds:
            seed_plan = PaperMiniGridIndependentSeeds(
                optimizer_seed=optimizer_seed,
                validation_seeds=resolved.validation_seeds,
                source_evaluation_seeds=resolved.source_evaluation_seeds,
                target_evaluation_seeds=resolved.target_evaluation_seeds,
                random_action_seeds=resolved.random_action_seeds,
            )
            outcome = evaluate_paper_minigrid_cell_independent(
                cell, seed_plan, resolved.ppo
            )
            runs.append(
                PaperMiniGridRobustnessRun(
                    cell_id=cell.cell_id,
                    optimizer_seed=optimizer_seed,
                    source_environment=cell.source.identifier,
                    target_environment=cell.target.identifier,
                    source_layout_seed=cell.source_seed,
                    target_layout_seed=cell.target_seed,
                    perturbation_type=cell.perturbation_type,
                    control_role=cell.control_role,
                    expected_transfer=cell.expected_transfer,
                    source_success_rate=outcome.source_success_rate,
                    target_zero_shot_success_rate=(
                        outcome.target_zero_shot_success_rate
                    ),
                    target_random_success_rate=outcome.target_random_success_rate,
                    source_efficiency_score=outcome.source_efficiency_score,
                    target_zero_shot_efficiency_score=(
                        outcome.target_zero_shot_efficiency_score
                    ),
                    target_random_efficiency_score=(
                        outcome.target_random_efficiency_score
                    ),
                    transfer_effect_score=outcome.transfer_effect_score,
                    source_mastered=outcome.source_mastered,
                    observed_transfer_success=outcome.observed_transfer_success,
                    agrees_with_reference=outcome.agrees_with_reference,
                    evaluation_episodes_per_arm=len(resolved.target_evaluation_seeds),
                )
            )
    return tuple(runs), _summarize_runs(runs, resolved_cells)


def write_paper_minigrid_robustness(
    output_dir: str | Path,
    config: PaperMiniGridRobustnessConfig | None = None,
    cells: tuple[PaperMiniGridCell, ...] | None = None,
) -> tuple[Path, Path, Path]:
    """Write tidy training-run rows, cell summaries, and a seed manifest."""

    resolved = config or PaperMiniGridRobustnessConfig()
    resolved_cells = cells or paper_minigrid_cells()
    runs, summaries = run_paper_minigrid_robustness(resolved, resolved_cells)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    runs_path = directory / "minigrid_robustness_runs.jsonl"
    summaries_path = directory / "minigrid_robustness_summaries.jsonl"
    manifest_path = directory / "minigrid_robustness_manifest.json"
    _write_jsonl(runs_path, runs)
    _write_jsonl(summaries_path, summaries)
    agreement_count = sum(int(row.agrees_with_reference) for row in runs)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "evidence_status": EVIDENCE_STATUS,
        "historical_protocol_modified": False,
        "paper_level_conclusion": False,
        "replication_unit": "independently_seeded_ppo_training_run",
        "config": asdict(resolved),
        "logical_cell_count": len(resolved_cells),
        "training_run_count": len(runs),
        "evaluation_episode_count": len(runs)
        * len(resolved.target_evaluation_seeds)
        * 3,
        "claim_checks": {
            "agreement_count": agreement_count,
            "all_training_runs_agree": agreement_count == len(runs),
            "majority_agree_cell_count": sum(
                int(row.agreement_rate > 0.5) for row in summaries
            ),
            "cell_count": len(summaries),
        },
        "artifacts": {
            "training_runs": runs_path.name,
            "cell_summaries": summaries_path.name,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return runs_path, summaries_path, manifest_path


def _summarize_runs(
    runs: list[PaperMiniGridRobustnessRun],
    cells: tuple[PaperMiniGridCell, ...],
) -> tuple[PaperMiniGridRobustnessSummary, ...]:
    summaries: list[PaperMiniGridRobustnessSummary] = []
    for cell in cells:
        selected = [row for row in runs if row.cell_id == cell.cell_id]
        if not selected:
            _raise_invalid(f"cell {cell.cell_id!r} has no robustness runs")
        agreements = sum(int(row.agrees_with_reference) for row in selected)
        lower, upper = _wilson_interval(agreements, len(selected))
        summaries.append(
            PaperMiniGridRobustnessSummary(
                cell_id=cell.cell_id,
                expected_transfer=cell.expected_transfer,
                training_run_count=len(selected),
                agreement_count=agreements,
                agreement_rate=agreements / len(selected),
                agreement_ci_lower=lower,
                agreement_ci_upper=upper,
                observed_transfer_count=sum(
                    int(row.observed_transfer_success) for row in selected
                ),
                mean_source_success_rate=fmean(
                    row.source_success_rate for row in selected
                ),
                mean_target_zero_shot_success_rate=fmean(
                    row.target_zero_shot_success_rate for row in selected
                ),
                mean_target_random_success_rate=fmean(
                    row.target_random_success_rate for row in selected
                ),
                mean_transfer_effect_score=fmean(
                    row.transfer_effect_score for row in selected
                ),
            )
        )
    return tuple(summaries)


def _wilson_interval(successes: int, count: int) -> tuple[float, float]:
    if count < 1 or not 0 <= successes <= count:
        _raise_invalid("Wilson interval counts are invalid")
    z = 1.959963984540054
    proportion = successes / count
    denominator = 1.0 + z * z / count
    center = (proportion + z * z / (2.0 * count)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / count + z * z / (4.0 * count * count)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _write_jsonl(path: Path, rows: tuple[Any, ...]) -> None:
    path.write_text(
        "".join(json.dumps(asdict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


__all__ = [
    "PaperMiniGridRobustnessConfig",
    "PaperMiniGridRobustnessRun",
    "PaperMiniGridRobustnessSummary",
    "run_paper_minigrid_robustness",
    "write_paper_minigrid_robustness",
]
