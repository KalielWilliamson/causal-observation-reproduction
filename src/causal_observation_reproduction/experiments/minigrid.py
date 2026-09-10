"""Portable MiniGrid source-target design and structural readiness experiment.

The pair matrix and MDP-to-SCM comparison are deterministic paper inputs.  A
separate optional policy evaluator may attach observed transfer outcomes to
these rows; this module never requires a tracker, cloud store, or scheduler.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, NoReturn

MiniGridFamily = Literal["Empty", "DoorKey"]
ExpectedRelation = Literal[
    "identity", "high_transfer", "scale_sensitive", "low_transfer"
]


def _raise_invalid(message: str) -> NoReturn:
    """Raise a single input-contract error from validation helpers."""

    raise ValueError(message)


@dataclass(frozen=True)
class MiniGridPairMatrixConfig:
    """Frozen local design factors for the external source-target panel."""

    families: tuple[MiniGridFamily, ...] = ("Empty",)
    sizes: tuple[int, ...] = (5, 8)
    seeds: tuple[int, ...] = (17,)
    include_same: bool = True
    include_scale: bool = True
    include_seed_shift: bool = True
    include_causal_negative_controls: bool = False
    max_pairs: int = 24
    transfer_threshold: float = 0.6

    def __post_init__(self) -> None:
        if not self.families:
            _raise_invalid("at least one MiniGrid family is required")
        if any(size < 3 for size in self.sizes):
            _raise_invalid("MiniGrid sizes must be at least three")
        if not self.seeds:
            _raise_invalid("at least one seed is required")
        if self.max_pairs < 1:
            _raise_invalid("max_pairs must be positive")
        if not 0.0 <= self.transfer_threshold <= 1.0:
            _raise_invalid("transfer threshold must lie in [0, 1]")


@dataclass(frozen=True)
class MiniGridPairSpec:
    """One predeclared source-target environment pair."""

    source_env_id: str
    target_env_id: str
    source_seed: int
    target_seed: int
    family: str
    perturbation_type: str
    expected_relation: ExpectedRelation
    source_family: str
    target_family: str

    @property
    def pair_id(self) -> str:
        """Return a stable public identifier for this exact pair definition."""

        payload = asdict(self)
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return f"minigrid-pair-{digest[:16]}"


@dataclass(frozen=True)
class StructuralReadiness:
    """Structural transfer-readiness score before any policy outcome is observed."""

    edge_jaccard_similarity: float
    state_count_ratio: float
    score: float
    band: Literal["low", "high"]


def generate_minigrid_pair_matrix(
    config: MiniGridPairMatrixConfig | None = None,
) -> tuple[MiniGridPairSpec, ...]:
    """Generate and balanced-limit the declared source-target pair matrix."""

    resolved = config or MiniGridPairMatrixConfig()
    families = tuple(dict.fromkeys(resolved.families))
    sizes = tuple(sorted(set(resolved.sizes)))
    seeds = tuple(sorted(set(resolved.seeds)))
    specs: list[MiniGridPairSpec] = []
    for family in families:
        environments = tuple(_environment_id(family, size) for size in sizes)
        if resolved.include_same:
            specs.extend(
                MiniGridPairSpec(
                    source_env_id=environment,
                    target_env_id=environment,
                    source_seed=seed,
                    target_seed=seed,
                    family=family,
                    perturbation_type="same_env_same_seed",
                    expected_relation="identity",
                    source_family=family,
                    target_family=family,
                )
                for environment in environments
                for seed in seeds
            )
        if resolved.include_seed_shift and len(seeds) > 1:
            specs.extend(
                MiniGridPairSpec(
                    source_env_id=environment,
                    target_env_id=environment,
                    source_seed=source_seed,
                    target_seed=target_seed,
                    family=family,
                    perturbation_type="same_env_seed_shift",
                    expected_relation="high_transfer",
                    source_family=family,
                    target_family=family,
                )
                for environment in environments
                for source_seed in seeds
                for target_seed in seeds
                if source_seed != target_seed
            )
        if resolved.include_scale and len(environments) > 1:
            specs.extend(
                MiniGridPairSpec(
                    source_env_id=source_environment,
                    target_env_id=target_environment,
                    source_seed=seed,
                    target_seed=seed,
                    family=family,
                    perturbation_type="same_family_scale_change",
                    expected_relation="scale_sensitive",
                    source_family=family,
                    target_family=family,
                )
                for source_environment in environments
                for target_environment in environments
                for seed in seeds
                if source_environment != target_environment
            )
    if resolved.include_causal_negative_controls:
        specs.extend(
            MiniGridPairSpec(
                source_env_id=_environment_id(source_family, size),
                target_env_id=_environment_id(target_family, size),
                source_seed=seed,
                target_seed=seed,
                family=f"{source_family}_to_{target_family}",
                perturbation_type="cross_family_causal_negative_control",
                expected_relation="low_transfer",
                source_family=source_family,
                target_family=target_family,
            )
            for source_family, target_family in _negative_control_pairs(families)
            for size in sizes
            for seed in seeds
        )
    unique = {spec.pair_id: spec for spec in specs}
    ordered = tuple(
        sorted(
            unique.values(),
            key=lambda spec: (
                _family_rank(spec.source_family),
                _family_rank(spec.target_family),
                spec.perturbation_type,
                spec.source_env_id,
                spec.target_env_id,
                spec.source_seed,
                spec.target_seed,
            ),
        )
    )
    return _balanced_limit(ordered, limit=resolved.max_pairs)


def project_pair_environment_to_scm(
    env_id: str, seed: int
) -> tuple[tuple[str, str], ...]:
    """Build the static MDP-to-SCM projection used before policy evaluation."""

    family, _ = _parse_environment_id(env_id)
    edges = {
        ("agent_pose_t", "agent_pose_t1"),
        ("action_t", "agent_pose_t1"),
        ("front_cell_blocked_t", "agent_pose_t1"),
        ("goal_region_t", "at_goal_t1"),
        ("agent_pose_t1", "at_goal_t1"),
    }
    if family == "DoorKey":
        edges.update(
            {
                ("action_t", "carrying_key_t1"),
                ("carrying_key_t", "carrying_key_t1"),
                ("action_t", "door_open_t1"),
                ("carrying_key_t", "door_open_t1"),
                ("door_open_t", "door_open_t1"),
                ("door_open_t", "at_goal_t1"),
            }
        )
    del seed
    return tuple(sorted(edges))


def structural_readiness(
    pair: MiniGridPairSpec, *, threshold: float = 0.6
) -> StructuralReadiness:
    """Compute the frozen SCM readiness score for a source-target pair."""

    source_edges = set(
        project_pair_environment_to_scm(pair.source_env_id, pair.source_seed)
    )
    target_edges = set(
        project_pair_environment_to_scm(pair.target_env_id, pair.target_seed)
    )
    similarity = len(source_edges & target_edges) / max(
        1, len(source_edges | target_edges)
    )
    source_nodes = {node for edge in source_edges for node in edge}
    target_nodes = {node for edge in target_edges for node in edge}
    ratio = min(len(source_nodes), len(target_nodes)) / max(
        len(source_nodes), len(target_nodes)
    )
    score = 0.7 * similarity + 0.3 * ratio
    return StructuralReadiness(
        edge_jaccard_similarity=similarity,
        state_count_ratio=ratio,
        score=score,
        band="high" if score >= threshold else "low",
    )


def write_minigrid_design(
    output_dir: str | Path, config: MiniGridPairMatrixConfig | None = None
) -> tuple[Path, Path]:
    """Write the frozen pair matrix and pre-policy structural readiness rows."""

    resolved = config or MiniGridPairMatrixConfig()
    pairs = generate_minigrid_pair_matrix(resolved)
    rows = [
        {
            **asdict(pair),
            "pair_id": pair.pair_id,
            **asdict(structural_readiness(pair, threshold=resolved.transfer_threshold)),
            "policy_evaluation_status": "requires_optional_policy_evaluator",
        }
        for pair in pairs
    ]
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rows_path = directory / "minigrid_pair_matrix.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest_path = directory / "minigrid_design_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "causal-observation-reproduction.minigrid.v1",
                "config": asdict(resolved),
                "pair_count": len(pairs),
                "policy_evaluation_status": "requires_optional_policy_evaluator",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return rows_path, manifest_path


def _environment_id(family: MiniGridFamily, size: int) -> str:
    return f"MiniGrid-{family}-{size}x{size}-v0"


def _parse_environment_id(env_id: str) -> tuple[MiniGridFamily, int]:
    parts = env_id.split("-")
    family = parts[1]
    size = int(parts[2].split("x", maxsplit=1)[0])
    if family == "Empty":
        return "Empty", size
    if family == "DoorKey":
        return "DoorKey", size
    message = f"unsupported MiniGrid family: {family}"
    _raise_invalid(message)


def _negative_control_pairs(
    families: tuple[MiniGridFamily, ...],
) -> tuple[tuple[MiniGridFamily, MiniGridFamily], ...]:
    if "Empty" in families and "DoorKey" in families:
        return (("Empty", "DoorKey"),)
    return tuple(
        (source, target)
        for index, source in enumerate(families)
        for target in families[index + 1 :]
    )


def _family_rank(family: str) -> tuple[int, str]:
    rank = {"Empty": 0, "DoorKey": 1}
    return rank.get(family, 2), family


def _balanced_limit(
    specs: tuple[MiniGridPairSpec, ...], *, limit: int
) -> tuple[MiniGridPairSpec, ...]:
    if len(specs) <= limit:
        return specs
    buckets: dict[str, list[MiniGridPairSpec]] = {}
    for spec in specs:
        buckets.setdefault(spec.perturbation_type, []).append(spec)
    selected: list[MiniGridPairSpec] = []
    indices = dict.fromkeys(sorted(buckets), 0)
    while len(selected) < limit:
        progressed = False
        for name in sorted(buckets):
            index = indices[name]
            if index < len(buckets[name]):
                selected.append(buckets[name][index])
                indices[name] += 1
                progressed = True
                if len(selected) == limit:
                    break
        if not progressed:
            break
    return tuple(selected)


__all__ = [
    "ExpectedRelation",
    "MiniGridFamily",
    "MiniGridPairMatrixConfig",
    "MiniGridPairSpec",
    "StructuralReadiness",
    "generate_minigrid_pair_matrix",
    "project_pair_environment_to_scm",
    "structural_readiness",
    "write_minigrid_design",
]
