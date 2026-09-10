"""Contracts for the one-command public experiment interface."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from causal_observation_reproduction.runner import (
    parse_args,
    run_benchmark,
    write_run_manifest,
)
from causal_observation_reproduction.specs import (
    EXTENSION_EXPERIMENTS,
    MANIFEST_SCHEMA_VERSION,
    RunSpec,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_default_command_is_the_exact_sequential_smoke_path() -> None:
    assert parse_args([]) == RunSpec()


def test_manifest_marks_the_manuscript_boundary(tmp_path: Path) -> None:
    spec = RunSpec(experiment="sequential", output_dir=tmp_path.as_posix())

    manifest_path = write_run_manifest(spec)

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert payload["scope"] == "manuscript"
    assert payload["experiment_contract"] == (
        "configs/reference/sequential_confirmation.json"
    )
    assert payload["run"] == {
        "experiment": "sequential",
        "name": "causal_observation_reproduction",
        "output_dir": tmp_path.as_posix(),
    }


def test_unknown_command_line_option_is_rejected() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--unreviewed-knob", "1"])


def test_extensions_are_unambiguously_labelled() -> None:
    assert EXTENSION_EXPERIMENTS
    assert all(name.startswith("extension-") for name in EXTENSION_EXPERIMENTS)
    parsed = parse_args(["extension-sequential-robustness"])
    assert parsed.experiment == "extension-sequential-robustness"


def test_minigrid_audit_dispatches_exact_paper_rows(tmp_path: Path) -> None:
    result = run_benchmark(
        RunSpec(experiment="minigrid-audit", output_dir=tmp_path.as_posix())
    )

    assert result.experiment == "minigrid-audit"
    assert result.primary_artifact == tmp_path / "minigrid_external_control.jsonl"
    assert all(path.is_file() for path in result.artifacts)
    assert len(result.primary_artifact.read_text(encoding="utf-8").splitlines()) == 16
