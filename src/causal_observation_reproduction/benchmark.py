"""Single dispatch surface for manuscript reproductions and named extensions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from causal_observation_reproduction.extensions import (
    run_certified_cache_control,
    run_minigrid_design,
    run_minigrid_policy,
    run_minigrid_robustness,
    run_one_step_strengthened,
    run_poscm_dqn,
    run_poscm_transfer,
    run_sequential_robustness,
)
from causal_observation_reproduction.reference.runners import (
    run_minigrid_reference,
    run_one_step_reference,
    run_paper_reference,
    run_sequential_reference,
    run_sequential_smoke,
)
from causal_observation_reproduction.specs import ExperimentName


@dataclass(frozen=True)
class BenchmarkRun:
    """Stable description of artifacts written by one public experiment."""

    experiment: ExperimentName
    artifacts: tuple[Path, ...]

    @property
    def primary_artifact(self) -> Path:
        """Return the main empirical artifact for the selected experiment."""

        return self.artifacts[0]


def run(experiment: ExperimentName, output_dir: str | Path) -> BenchmarkRun:
    """Run one exact manuscript experiment or explicitly labelled extension."""

    directory = Path(output_dir)
    if experiment == "smoke":
        artifacts = run_sequential_smoke(directory)
    elif experiment == "paper":
        artifacts = run_paper_reference(directory)
    elif experiment == "sequential":
        artifacts = run_sequential_reference(directory)
    elif experiment == "one-step":
        artifacts = run_one_step_reference(directory)
    elif experiment == "minigrid-audit":
        artifacts = run_minigrid_reference(directory)
    elif experiment == "extension-certified-cache-control":
        artifacts = run_certified_cache_control(directory)
    elif experiment == "extension-sequential-robustness":
        artifacts = run_sequential_robustness(directory)
    elif experiment == "extension-one-step-strengthened":
        artifacts = run_one_step_strengthened(directory)
    elif experiment == "extension-poscm-transfer":
        artifacts = run_poscm_transfer(directory)
    elif experiment == "extension-poscm-dqn":
        artifacts = run_poscm_dqn(directory)
    elif experiment == "extension-minigrid-design":
        artifacts = run_minigrid_design(directory)
    elif experiment == "extension-minigrid-policy":
        artifacts = run_minigrid_policy(directory)
    else:
        artifacts = run_minigrid_robustness(directory)
    return BenchmarkRun(experiment=experiment, artifacts=artifacts)


__all__ = ["BenchmarkRun", "ExperimentName", "run"]
