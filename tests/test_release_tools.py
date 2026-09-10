"""Tests for local-only public release preparation."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "release_tools", Path(__file__).parents[1] / "tools" / "release_tools.py"
)
assert _SPEC is not None
assert _SPEC.loader is not None
_TOOLS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TOOLS)


def test_evidence_verifier_fails_closed_on_a_missing_bundle(tmp_path: Path) -> None:
    source = tmp_path / "paper"
    source.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source_revision": "a" * 40,
                "required_evidence_files": ["sequential/parity_report.json"],
                "protocol_sha256": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing"):
        _TOOLS.verify_evidence(source, manifest)


def test_evidence_packager_copies_the_complete_verified_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "paper"
    nested = source / "sequential"
    nested.mkdir(parents=True)
    (nested / "rows.json").write_text("[]\n", encoding="utf-8")
    verified: dict[str, Any] = {
        "source_revision": "a" * 40,
        "sequential_episode_count": 152_064,
        "one_step_row_count": 274,
        "minigrid_row_count": 16,
        "evidence_sha256": "b" * 64,
        "status": "verified_not_yet_deposited",
    }
    monkeypatch.setattr(_TOOLS, "verify_evidence", lambda *_args: verified)

    destination = tmp_path / "package"
    result = _TOOLS.package_evidence(source, destination, tmp_path / "unused.json")

    assert result["evidence_sha256"] == "b" * 64
    assert (destination / "sequential" / "rows.json").is_file()
    assert (destination / "evidence_manifest.json").is_file()


def test_snapshot_exports_a_tree_without_git_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init"], cwd=repository, check=True, capture_output=True)
    (repository / "included.txt").write_text("included\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "included.txt"], cwd=repository, check=True, capture_output=True
    )
    tree = subprocess.run(
        ["git", "write-tree"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.chdir(repository)

    snapshot = _TOOLS.export_public_snapshot(tree, tmp_path / "snapshot")

    provenance = (snapshot / "PUBLIC_SNAPSHOT.md").read_text("utf-8")
    assert tree in provenance
    assert (snapshot / "included.txt").is_file()
    assert not (snapshot / ".git").exists()


def test_snapshot_scan_checks_paths_as_well_as_contents(tmp_path: Path) -> None:
    destination = tmp_path / "snapshot"
    forbidden = destination / ("sequential_" + "v" + "2.py")
    forbidden.parent.mkdir()
    forbidden.write_text("clean content", encoding="utf-8")

    with pytest.raises(ValueError, match="forbidden path"):
        _TOOLS._reject_forbidden_public_terms(destination)


def test_snapshot_scan_rejects_credential_like_values(tmp_path: Path) -> None:
    destination = tmp_path / "snapshot"
    destination.mkdir()
    (destination / "settings.txt").write_text(
        "access_" + "token=" + "0123456789abcdefghijklmnopqrstuvwxyz\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="credential-like assignment"):
        _TOOLS._reject_forbidden_public_terms(destination)


def test_worktree_snapshot_includes_only_current_nonignored_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init"], cwd=repository, check=True, capture_output=True)
    (repository / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    (repository / "modified.txt").write_text("before\n", encoding="utf-8")
    (repository / "deleted.txt").write_text("deleted\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", ".gitignore", "modified.txt", "deleted.txt"],
        cwd=repository,
        check=True,
    )
    (repository / "modified.txt").write_text("after\n", encoding="utf-8")
    (repository / "deleted.txt").unlink()
    (repository / "untracked.txt").write_text("included\n", encoding="utf-8")
    (repository / "ignored.txt").write_text("excluded\n", encoding="utf-8")
    monkeypatch.chdir(repository)

    snapshot = _TOOLS.export_public_worktree_snapshot(tmp_path / "snapshot")

    assert (snapshot / "modified.txt").read_text("utf-8") == "after\n"
    assert (snapshot / "untracked.txt").is_file()
    assert not (snapshot / "deleted.txt").exists()
    assert not (snapshot / "ignored.txt").exists()
    assert not (snapshot / ".git").exists()
