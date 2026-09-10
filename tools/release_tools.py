"""Prepare a history-free snapshot and verify the exact manuscript bundle."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import asdict
from importlib.resources import files
from pathlib import Path
from typing import Any, cast

from causal_observation_reproduction.reference.parity import (
    verify_locked_sequential_confirmation,
    verify_one_step_calibration,
    verify_online_compute_counts,
)
from causal_observation_reproduction.reference.sequential_experiment import (
    ConditionalEfficiencyConfirmationResult,
)

FORBIDDEN_PUBLIC_TERMS = ("cog" + "trace", "sequential_" + "v" + "2")
FORBIDDEN_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private key",
        re.compile(r"-----BEGIN (?:EC |OPENSSH |RSA )?PRIVATE KEY-----"),
    ),
    ("GitHub token", re.compile(r"\bgh(?:p|o|u|s|r)_[A-Za-z0-9]{30,}\b")),
    ("GitHub token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,}\b")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    (
        "credential-like assignment",
        re.compile(
            r"(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|"
            r"client[_-]?secret)\b\s*[:=]\s*(?!\$\{\{)[\"']?[A-Za-z0-9_./+=-]{16,}"
        ),
    ),
)
_NON_TEXT_SUFFIXES = {
    ".dll",
    ".exe",
    ".gz",
    ".parquet",
    ".pdf",
    ".png",
    ".pt",
    ".so",
    ".whl",
    ".zip",
}


def _load_json(path: Path) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))


def _load_rows(path: Path) -> tuple[dict[str, Any], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(
        isinstance(row, dict) for row in payload
    ):
        message = f"expected a JSON row array: {path}"
        raise ValueError(message)
    return tuple(cast("list[dict[str, Any]]", payload))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "evidence_manifest.json":
            continue
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def verify_evidence(
    source: Path, manifest_path: Path = Path("release/manifest.json")
) -> dict[str, Any]:
    """Independently verify a complete `paper` output directory."""

    manifest = _load_json(manifest_path)
    for relative in manifest["required_evidence_files"]:
        path = source / str(relative)
        if not path.is_file():
            message = f"required evidence file is missing: {relative}"
            raise ValueError(message)

    run_manifest = _load_json(source / "run_manifest.json")
    if (
        run_manifest.get("scope") != "manuscript"
        or run_manifest.get("run", {}).get("experiment") != "paper"
    ):
        message = "evidence source is not a full paper run"
        raise ValueError(message)

    expected_revision = str(manifest["source_revision"])
    sequential_dir = source / "sequential"
    rows = _load_rows(sequential_dir / "confirmation_rows.json")
    aggregate = _load_rows(sequential_dir / "confirmation_metrics.json")
    result = ConditionalEfficiencyConfirmationResult(
        rows=rows,
        aggregate=aggregate,
        models={},
        artifacts={
            "rows": str(sequential_dir / "confirmation_rows.json"),
            "manifest": str(sequential_dir / "locked_confirmation_manifest.json"),
            "paper_projection": str(
                sequential_dir / "conditional_efficiency_confirmation_summary.json"
            ),
            "clustered_noninferiority": str(
                sequential_dir / "clustered_noninferiority.json"
            ),
        },
    )
    parity = verify_locked_sequential_confirmation(result)
    if parity.source_revision != expected_revision or len(rows) != 152_064:
        message = "sequential evidence scope or source revision differs"
        raise ValueError(message)
    tracked_parity = _load_json(sequential_dir / "parity_report.json")
    if tracked_parity.get("passed") is not True:
        message = "sequential parity report did not pass"
        raise ValueError(message)
    compute = _load_json(sequential_dir / "online_compute_characterization.json")
    verify_online_compute_counts(compute)

    one_step_manifest = _load_json(
        source / "one-step" / "datasets" / "dataset_manifest.json"
    )
    if one_step_manifest.get("source_revision") != expected_revision:
        message = "one-step evidence source revision differs"
        raise ValueError(message)
    if int(one_step_manifest.get("row_count", -1)) != 274:
        message = "one-step evidence must contain 274 total rows"
        raise ValueError(message)
    one_step_parity = verify_one_step_calibration(
        dataset_manifest_path=source
        / "one-step"
        / "datasets"
        / "dataset_manifest.json",
        split_manifest_path=source / "one-step" / "splits" / "split_manifest.json",
        policy_metrics_path=source / "one-step" / "metrics" / "policy_metrics.csv",
    )
    tracked_one_step_parity = _load_json(source / "one-step" / "parity_report.json")
    if tracked_one_step_parity != asdict(one_step_parity):
        message = "one-step parity report differs from independently computed results"
        raise ValueError(message)

    minigrid_dir = source / "minigrid-audit"
    source_rows = files("causal_observation_reproduction.reference").joinpath(
        "data/minigrid_external_control.jsonl"
    )
    if (minigrid_dir / "minigrid_external_control.jsonl").read_bytes() != (
        source_rows.read_bytes()
    ):
        message = "MiniGrid rows differ from the checked paper artifact"
        raise ValueError(message)
    minigrid_summary = _load_json(
        minigrid_dir / "minigrid_external_control_summary.json"
    )
    if minigrid_summary.get("row_count") != 16:
        message = "MiniGrid audit must contain 16 evaluated rows"
        raise ValueError(message)

    expected_hashes = manifest["protocol_sha256"]
    root = manifest_path.parent.parent
    actual_hashes = {
        "sequential": _sha256(root / "configs/reference/sequential_confirmation.json"),
        "one_step": _sha256(root / "configs/reference/one_step_gate.json"),
        "source_provenance": _sha256(root / "provenance/source_manifest.json"),
    }
    if actual_hashes != expected_hashes:
        message = "tracked protocol or provenance hashes differ"
        raise ValueError(message)

    return {
        "source_revision": expected_revision,
        "sequential_episode_count": len(rows),
        "one_step_row_count": 274,
        "one_step_assessment": one_step_parity.assessment,
        "one_step_corpus_cardinality_matches": (
            one_step_parity.corpus_cardinality_matches
        ),
        "minigrid_row_count": 16,
        "evidence_sha256": _tree_sha256(source),
        "status": "verified_not_yet_deposited",
    }


def package_evidence(
    source: Path,
    destination: Path,
    manifest_path: Path = Path("release/manifest.json"),
) -> dict[str, Any]:
    """Copy a verified full paper bundle to an empty archival directory."""

    verified = verify_evidence(source, manifest_path)
    _require_empty_destination(destination)
    shutil.copytree(source, destination, dirs_exist_ok=True)
    bundle_manifest = {
        "schema_version": "causal-observation-reproduction.evidence-bundle.v1",
        **verified,
    }
    (destination / "evidence_manifest.json").write_text(
        json.dumps(bundle_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return bundle_manifest


def export_public_snapshot(ref: str, destination: Path) -> Path:
    """Export one Git tree without history, refusing an unsafe destination."""

    _require_empty_destination(destination)
    archive = subprocess.run(
        ["git", "archive", "--format=tar", ref], check=True, capture_output=True
    ).stdout
    destination.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        tar.extractall(destination, filter="data")
    _reject_forbidden_public_terms(destination)
    revision = subprocess.run(
        ["git", "rev-parse", ref], check=True, capture_output=True, text=True
    ).stdout.strip()
    (destination / "PUBLIC_SNAPSHOT.md").write_text(
        "# Public snapshot provenance\n\n"
        f"Exported from Git tree `{revision}` without Git history.\n"
        "Initialize a new repository here only after completing "
        "`docs/public_release.md`.\n",
        encoding="utf-8",
    )
    return destination


def export_public_worktree_snapshot(destination: Path) -> Path:
    """Export the reviewed working tree without history or ignored files."""

    _require_empty_destination(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-staging-", dir=destination.parent)
    )
    root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    ).resolve()
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        check=True,
        capture_output=True,
    ).stdout
    try:
        for encoded_relative in listed.split(b"\0"):
            if not encoded_relative:
                continue
            relative, source = _resolve_worktree_source(root, encoded_relative)
            # Cached paths deleted in the working tree are intentionally omitted.
            if not source.is_file():
                continue
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        _reject_forbidden_public_terms(staging)
        (staging / "PUBLIC_SNAPSHOT.md").write_text(
            "# Public snapshot provenance\n\n"
            "Exported from the reviewed working tree without Git history. Tracked "
            "modifications and non-ignored untracked files are included; tracked "
            "deletions are omitted.\n",
            encoding="utf-8",
        )
        if destination.exists():
            destination.rmdir()
        staging.replace(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def _require_empty_destination(destination: Path) -> None:
    if destination.exists() and any(destination.iterdir()):
        message = f"destination must not already contain files: {destination}"
        raise ValueError(message)


def _resolve_worktree_source(root: Path, encoded_relative: bytes) -> tuple[Path, Path]:
    relative = Path(os.fsdecode(encoded_relative))
    if relative.is_absolute() or ".." in relative.parts:
        message = f"unsafe path returned by Git: {relative}"
        raise ValueError(message)
    source = (root / relative).resolve()
    if not source.is_relative_to(root):
        message = f"source path resolves outside the repository: {relative}"
        raise ValueError(message)
    return relative, source


def _reject_forbidden_public_terms(destination: Path) -> None:
    for path in destination.rglob("*"):
        relative = path.relative_to(destination).as_posix().lower()
        if any(term in relative for term in FORBIDDEN_PUBLIC_TERMS):
            message = f"public snapshot contains a forbidden path: {path}"
            raise ValueError(message)
        if not path.is_file() or path.suffix.lower() in _NON_TEXT_SUFFIXES:
            continue
        content = path.read_text("utf-8", errors="ignore")
        if any(term in content.lower() for term in FORBIDDEN_PUBLIC_TERMS):
            message = f"public snapshot contains a forbidden term: {path}"
            raise ValueError(message)
        for label, pattern in FORBIDDEN_SECRET_PATTERNS:
            if pattern.search(content):
                message = f"public snapshot contains a {label}: {path}"
                raise ValueError(message)


def main() -> None:
    """Run the explicitly requested local release-preparation command."""

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    snapshot = subparsers.add_parser("snapshot")
    snapshot.add_argument("--ref", default="HEAD")
    snapshot.add_argument("--destination", type=Path, required=True)
    worktree_snapshot = subparsers.add_parser("snapshot-worktree")
    worktree_snapshot.add_argument("--destination", type=Path, required=True)
    evidence = subparsers.add_parser("package-evidence")
    evidence.add_argument("--source", type=Path, required=True)
    evidence.add_argument("--destination", type=Path, required=True)
    evidence.add_argument(
        "--manifest", type=Path, default=Path("release/manifest.json")
    )
    verify = subparsers.add_parser("verify-evidence")
    verify.add_argument("--source", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, default=Path("release/manifest.json"))
    args = parser.parse_args()
    if args.command == "snapshot":
        result: Any = {
            "snapshot": str(export_public_snapshot(args.ref, args.destination))
        }
    elif args.command == "snapshot-worktree":
        result = {"snapshot": str(export_public_worktree_snapshot(args.destination))}
    elif args.command == "package-evidence":
        result = package_evidence(args.source, args.destination, args.manifest)
    else:
        result = verify_evidence(args.source, args.manifest)
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
