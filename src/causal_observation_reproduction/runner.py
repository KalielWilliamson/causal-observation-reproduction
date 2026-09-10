"""One command-line interface for every public reproduction entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from causal_observation_reproduction.benchmark import BenchmarkRun, run
from causal_observation_reproduction.specs import (
    EXPERIMENTS,
    RunSpec,
    experiment_name,
    run_manifest,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def write_run_manifest(spec: RunSpec) -> Path:
    """Write resolved public launch metadata and return its path."""

    output_dir = Path(spec.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(run_manifest(spec), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def run_benchmark(spec: RunSpec) -> BenchmarkRun:
    """Execute the selected experiment through the sole public dispatch API."""

    return run(spec.experiment, spec.output_dir)


def parse_args(args: Sequence[str] | None = None) -> RunSpec:
    """Parse the dependency-free public CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment",
        nargs="?",
        choices=EXPERIMENTS,
        default="smoke",
        help="Experiment to run (default: smoke). Use 'paper' for the full reproduction.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Artifact directory (default: artifacts/runs/<experiment>).",
    )
    parser.add_argument(
        "--name",
        default="causal_observation_reproduction",
        help="Run name recorded in the reproducibility manifest.",
    )
    parsed = parser.parse_args(args)
    experiment = experiment_name(str(parsed.experiment))
    output_dir = str(parsed.output_dir or f"artifacts/runs/{experiment}")
    return RunSpec(
        experiment=experiment,
        output_dir=output_dir,
        name=str(parsed.name),
    )


def main(args: Sequence[str] | None = None) -> None:
    """Execute one experiment and print a machine-readable artifact summary."""

    spec = parse_args(args)
    manifest_path = write_run_manifest(spec)
    result = run_benchmark(spec)
    sys.stdout.write(
        json.dumps(
            {
                "experiment": result.experiment,
                "manifest": str(manifest_path),
                "artifacts": [str(path) for path in result.artifacts],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
