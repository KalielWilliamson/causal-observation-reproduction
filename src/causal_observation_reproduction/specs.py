"""Strongly typed launch contract for the single public experiment API."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

MANIFEST_SCHEMA_VERSION = "causal-observation-reproduction.manifest.v1"

ExperimentName = Literal[
    "smoke",
    "paper",
    "sequential",
    "one-step",
    "minigrid-audit",
    "extension-certified-cache-control",
    "extension-sequential-robustness",
    "extension-one-step-strengthened",
    "extension-poscm-transfer",
    "extension-poscm-dqn",
    "extension-minigrid-design",
    "extension-minigrid-policy",
    "extension-minigrid-robustness",
]

MANUSCRIPT_EXPERIMENTS: tuple[ExperimentName, ...] = (
    "paper",
    "sequential",
    "one-step",
    "minigrid-audit",
)
EXTENSION_EXPERIMENTS: tuple[ExperimentName, ...] = (
    "extension-certified-cache-control",
    "extension-sequential-robustness",
    "extension-one-step-strengthened",
    "extension-poscm-transfer",
    "extension-poscm-dqn",
    "extension-minigrid-design",
    "extension-minigrid-policy",
    "extension-minigrid-robustness",
)
EXPERIMENTS: tuple[ExperimentName, ...] = (
    "smoke",
    *MANUSCRIPT_EXPERIMENTS,
    *EXTENSION_EXPERIMENTS,
)


@dataclass(frozen=True)
class RunSpec:
    """Resolved experiment and destination recorded for every public run."""

    experiment: ExperimentName = "smoke"
    output_dir: str = "artifacts/runs/smoke"
    name: str = "causal_observation_reproduction"


def experiment_name(value: str) -> ExperimentName:
    """Validate and narrow a user-supplied experiment name."""

    if value not in EXPERIMENTS:
        choices = ", ".join(EXPERIMENTS)
        message = f"experiment must be one of: {choices}"
        raise ValueError(message)
    return value


def run_manifest(spec: RunSpec) -> dict[str, Any]:
    """Materialize launch metadata and the manuscript/extension boundary."""

    contracts = {
        "smoke": "reduced structural check of the exact sequential implementation",
        "paper": "all executable experiments described in the manuscript",
        "sequential": "configs/reference/sequential_confirmation.json",
        "one-step": "configs/reference/one_step_gate.json",
        "minigrid-audit": "packaged 16-row external-control appendix artifact",
        "extension-certified-cache-control": "post-manuscript certified cache-control application",
        "extension-sequential-robustness": "post hoc history-dependent sequential robustness design",
        "extension-one-step-strengthened": "larger independent-test one-step robustness design",
        "extension-poscm-transfer": "additional POSCM transfer design",
        "extension-poscm-dqn": "additional local-DQN policy evaluation",
        "extension-minigrid-design": "additional synthetic MiniGrid design",
        "extension-minigrid-policy": "additional local-PPO policy evaluation",
        "extension-minigrid-robustness": "additional independent-seed PPO robustness run",
    }
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run": asdict(spec),
        "scope": (
            "manuscript"
            if spec.experiment in MANUSCRIPT_EXPERIMENTS
            else "smoke"
            if spec.experiment == "smoke"
            else "extension"
        ),
        "experiment_contract": contracts[spec.experiment],
    }


__all__ = [
    "EXPERIMENTS",
    "EXTENSION_EXPERIMENTS",
    "MANIFEST_SCHEMA_VERSION",
    "MANUSCRIPT_EXPERIMENTS",
    "ExperimentName",
    "RunSpec",
    "experiment_name",
    "run_manifest",
]
