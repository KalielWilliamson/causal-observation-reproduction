"""Thin public runners around the pinned manuscript implementation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from causal_observation_reproduction.reference.compute_characterization import (
    characterize_locked_confirmation,
)
from causal_observation_reproduction.reference.minigrid_audit import (
    write_minigrid_external_control,
)
from causal_observation_reproduction.reference.parity import (
    verify_locked_sequential_confirmation,
    verify_one_step_calibration,
    verify_online_compute_counts,
)
from causal_observation_reproduction.reference.sequential_experiment import (
    ConditionalEfficiencyConfirmationConfig,
    run_conditional_efficiency_confirmation,
)


def run_sequential_smoke(output_dir: str | Path) -> tuple[Path, ...]:
    """Run a small structural check through the exact sequential implementation."""

    destination = Path(output_dir)
    config = ConditionalEfficiencyConfirmationConfig(
        training_topology_families=("positive_boundary_advantage",),
        heldout_topology_families=("qualifying_or_negative_boundary_scope",),
        regimes=("positive", "null"),
        distractor_counts=(0,),
        horizons=(3,),
        probe_costs=(0.1,),
        generic_rollout_budgets=(1, 16),
        primary_generic_rollout_budget=16,
        topology_instances_per_family=2,
        cluster_bootstrap_repetitions=50,
        training_seeds=(101, 102),
        evaluation_seeds=(401, 402),
        output_dir=str(destination),
    )
    result = run_conditional_efficiency_confirmation(config=config)
    return tuple(Path(path) for path in result.artifacts.values())


def run_sequential_reference(output_dir: str | Path) -> tuple[Path, ...]:
    """Run the full locked sequential protocol and fail on manuscript divergence."""

    destination = Path(output_dir)
    config = replace(
        ConditionalEfficiencyConfirmationConfig(), output_dir=str(destination)
    )
    result = run_conditional_efficiency_confirmation(config=config)
    parity_path = destination / "parity_report.json"
    verify_locked_sequential_confirmation(result, output_path=parity_path)
    compute_path = destination / "online_compute_characterization.json"
    compute = characterize_locked_confirmation(
        rows_path=Path(result.artifacts["rows"]),
        manifest_path=Path(result.artifacts["manifest"]),
        output_path=compute_path,
    )
    verify_online_compute_counts(compute)
    return (
        *(Path(path) for path in result.artifacts.values()),
        parity_path,
        compute_path,
    )


def run_one_step_reference(output_dir: str | Path) -> tuple[Path, ...]:
    """Run the exact source-derived one-step learned-gate experiment."""

    # This import is intentionally lazy: only this experiment needs the learned
    # extra (Pandas, PyArrow, Matplotlib, and Torch).
    from causal_observation_reproduction.reference.learned_coo_gate import (
        LearnedCOOGateConfig,
        LearnedCOOGateExperiment,
    )

    destination = Path(output_dir)
    experiment = LearnedCOOGateExperiment(
        replace(LearnedCOOGateConfig(), output_dir=str(destination))
    )
    experiment.run()
    parity_path = destination / "parity_report.json"
    verify_one_step_calibration(
        dataset_manifest_path=destination / "datasets" / "dataset_manifest.json",
        split_manifest_path=destination / "splits" / "split_manifest.json",
        policy_metrics_path=destination / "metrics" / "policy_metrics.csv",
        output_path=parity_path,
    )
    return (
        destination / "learned_coo_claim_evidence.json",
        destination / "metrics" / "policy_metrics.csv",
        destination / "datasets" / "learned_coo_examples.parquet",
        destination / "manifests" / "run_manifest.json",
        destination / "learned_coo_gate_report.md",
        parity_path,
    )


def run_minigrid_reference(output_dir: str | Path) -> tuple[Path, ...]:
    """Export and recompute the manuscript's exact 16-row MiniGrid audit."""

    return write_minigrid_external_control(output_dir)


def run_paper_reference(output_dir: str | Path) -> tuple[Path, ...]:
    """Run every executable manuscript experiment under one directory."""

    destination = Path(output_dir)
    return (
        *run_sequential_reference(destination / "sequential"),
        *run_one_step_reference(destination / "one-step"),
        *run_minigrid_reference(destination / "minigrid-audit"),
    )


__all__ = [
    "run_minigrid_reference",
    "run_one_step_reference",
    "run_paper_reference",
    "run_sequential_reference",
    "run_sequential_smoke",
]
