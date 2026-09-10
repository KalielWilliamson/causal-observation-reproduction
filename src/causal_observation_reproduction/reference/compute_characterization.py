"""Post-hoc, source-defined online-compute characterization for COO.

This module deliberately reports counted work and fixed-host CPU timing rather
than FLOPs.  The deployed COO gate is a small Python threshold table, while
the bounded comparator is dominated by simulator and interpreter work; FLOPs
would not describe either execution path faithfully.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from statistics import fmean, median
from typing import Any, cast

from causal_observation_reproduction.reference.quotient_family import (
    generate_causal_quotient_family,
)
from causal_observation_reproduction.reference.sequential_experiment import (
    BoundedRolloutVoiPolicyAdapter,
    ConditionalEfficiencyConfirmationConfig,
    GateModel,
    GatePolicyAdapter,
    InformationPolicy,
    PolicyName,
    Regime,
    _controller_features,
    _public_rollout,
    build_confirmation_specs,
)
from causal_observation_reproduction.reference.sequential_sensing import (
    SequentialQuotientSensingEnv,
)

SCHEMA_VERSION = (
    "causal_observation_reproduction.causal_quotient_online_compute_characterization.v1"
)


@dataclass(frozen=True)
class ComputeCharacterizationConfig:
    """Selection and timing policy for a descriptive replay."""

    primary_rollout_budget: int = 16
    warmup_repetitions: int = 2
    measured_repetitions: int = 5
    gate_batch_size: int = 10_000


def characterize_locked_confirmation(
    *,
    rows_path: Path,
    manifest_path: Path,
    output_path: Path | None = None,
    config: ComputeCharacterizationConfig | None = None,
) -> dict[str, Any]:
    """Characterize the locked positive budget-16 comparison without training.

    The original rows remain the outcome record.  This routine only rebuilds
    deterministic fixture metadata to count source-defined execution work and
    runs warm-up/repeated fixed-host elapsed timing of one representative decision per
    arm.  It excludes the shared controller, environment action selection,
    training, and representation construction.
    """
    resolved = config or ComputeCharacterizationConfig()
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    confirmation = _confirmation_config(dict(manifest["config"]))
    gate_rows = [
        row
        for row in rows
        if row["policy"] == "learned_causal_quotient"
        and row["regime"] == "positive"
        and int(row["generic_rollout_budget"]) == 0
    ]
    voi_rows = [
        row
        for row in rows
        if row["policy"] == "generic_bounded_rollout_voi"
        and row["regime"] == "positive"
        and int(row["generic_rollout_budget"]) == resolved.primary_rollout_budget
    ]
    pairs = _pair_rows(gate_rows, voi_rows)
    if not pairs:
        raise ValueError(
            "no matched positive gate/VoI rows at requested rollout budget"
        )
    gate_model = _gate_model(pairs[0][0])
    summaries = {
        "learned_causal_quotient": _gate_work([left for left, _ in pairs]),
        "generic_bounded_rollout_voi": _voi_work([right for _, right in pairs]),
    }
    representative = _representative_timing(
        pair=pairs[0],
        confirmation=confirmation,
        gate_model=gate_model,
        config=resolved,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "evidence_role": "post_hoc_descriptive_online_compute_characterization",
        "source": {
            "confirmation_rows_sha256": sha256(rows_path.read_bytes()).hexdigest(),
            "locked_manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
            "paired_positive_episodes": len(pairs),
            "primary_rollout_budget": resolved.primary_rollout_budget,
        },
        "scope": {
            "included": [
                "information-policy decision logic",
                "public rollout simulation",
            ],
            "excluded": [
                "shared base controller",
                "environment-action selection",
                "gate training",
                "representation construction",
            ],
            "not_claimed": [
                "FLOPs",
                "hardware-independent efficiency",
                "registered inferential result",
            ],
        },
        "counted_online_work": summaries,
        "cpu_decision_timing": representative,
        "host": {"python": sys.version.split()[0], "platform": platform.platform()},
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return payload


def _confirmation_config(
    raw: Mapping[str, Any],
) -> ConditionalEfficiencyConfirmationConfig:
    defaults = ConditionalEfficiencyConfirmationConfig()

    def sequence(key: str, default: tuple[Any, ...]) -> tuple[Any, ...]:
        value = raw.get(key, default)
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"confirmation config field {key!r} must be a sequence")
        return tuple(value)

    raw_regimes = tuple(str(value) for value in sequence("regimes", defaults.regimes))
    if any(value not in {"positive", "null", "invalid"} for value in raw_regimes):
        raise ValueError("confirmation config contains an unknown regime")
    regimes = cast("tuple[Regime, ...]", raw_regimes)

    raw_policies = tuple(
        str(value) for value in sequence("policy_arms", defaults.policy_arms)
    )
    allowed_policies = {
        "learned_causal_quotient",
        "generic_bounded_rollout_voi",
        "capacity_matched_no_quotient",
        "never_refine",
        "always_refine",
        "random_budget_matched",
        "uncertainty",
        "information_gain",
        "oracle",
    }
    if any(value not in allowed_policies for value in raw_policies):
        raise ValueError("confirmation config contains an unknown policy arm")
    policy_arms = cast("tuple[PolicyName, ...]", raw_policies)

    return ConditionalEfficiencyConfirmationConfig(
        training_topology_families=tuple(
            str(value)
            for value in sequence(
                "training_topology_families", defaults.training_topology_families
            )
        ),
        heldout_topology_families=tuple(
            str(value)
            for value in sequence(
                "heldout_topology_families", defaults.heldout_topology_families
            )
        ),
        regimes=regimes,
        distractor_counts=tuple(
            int(value)
            for value in sequence("distractor_counts", defaults.distractor_counts)
        ),
        horizons=tuple(int(value) for value in sequence("horizons", defaults.horizons)),
        probe_costs=tuple(
            float(value) for value in sequence("probe_costs", defaults.probe_costs)
        ),
        generic_rollout_budgets=tuple(
            int(value)
            for value in sequence(
                "generic_rollout_budgets", defaults.generic_rollout_budgets
            )
        ),
        primary_generic_rollout_budget=int(
            raw.get(
                "primary_generic_rollout_budget",
                defaults.primary_generic_rollout_budget,
            )
        ),
        topology_instances_per_family=int(
            raw.get(
                "topology_instances_per_family",
                defaults.topology_instances_per_family,
            )
        ),
        noninferiority_margin=float(
            raw.get("noninferiority_margin", defaults.noninferiority_margin)
        ),
        cluster_bootstrap_repetitions=int(
            raw.get(
                "cluster_bootstrap_repetitions",
                defaults.cluster_bootstrap_repetitions,
            )
        ),
        policy_arms=policy_arms,
        cache_policy=str(raw.get("cache_policy", defaults.cache_policy)),
        training_seeds=tuple(
            int(value) for value in sequence("training_seeds", defaults.training_seeds)
        ),
        evaluation_seeds=tuple(
            int(value)
            for value in sequence("evaluation_seeds", defaults.evaluation_seeds)
        ),
        output_dir=str(raw.get("output_dir", defaults.output_dir)),
    )


def _pair_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(
        row[key]
        for key in (
            "family_id",
            "seed",
            "distractor_count",
            "horizon",
            "probe_cost",
            "regime",
        )
    )


def _pair_rows(
    gate_rows: Sequence[Mapping[str, Any]], voi_rows: Sequence[Mapping[str, Any]]
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    voi_by_key = {_pair_key(row): row for row in voi_rows}
    pairs = [
        (row, voi_by_key[_pair_key(row)])
        for row in gate_rows
        if _pair_key(row) in voi_by_key
    ]
    if len(pairs) != len(gate_rows) or len(pairs) != len(voi_rows):
        raise ValueError("locked primary rows are not one-to-one paired")
    return sorted(pairs, key=lambda pair: _pair_key(pair[0]))


def _gate_model(row: Mapping[str, Any]) -> GateModel:
    model = dict(dict(row["policy_metadata"])["model"])
    entries = tuple(
        (bool(feature), float(cost), float(gain))
        for feature, cost, gain in model["gain_by_feature_cost"]
    )
    return GateModel(
        positive_gain=fmean(gain for feature, _, gain in entries if feature),
        nonpositive_gain=fmean(gain for feature, _, gain in entries if not feature),
        uses_quotient=bool(model["uses_query_feature"]),
        gain_by_feature_cost=entries,
    )


def _gate_work(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    # Gate features are stationary within an episode.  If it probes, the first
    # lookup selects it and every later decision short-circuits; otherwise it
    # evaluates its threshold once per eligible control step.
    calls = sum(1 if int(row["probe_count"]) else int(row["horizon"]) for row in rows)
    return {
        "decision_invocations": sum(int(row["horizon"]) for row in rows),
        "threshold_table_lookups": calls,
        "scalar_threshold_comparisons": calls,
        "public_rollouts": 0,
        "simulated_control_transitions": 0,
        "candidate_branches": 0,
    }


def _voi_work(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    transitions = 0
    scalar_elements = 0
    for row in rows:
        # The fixture factory fixes node_count=6+distractors, while the public
        # evaluator observation appends two scalars (step and horizon).
        width = 8 + int(row["distractor_count"])
        rollouts = int(row["rollout_count"])
        horizon = int(row["horizon"])
        transitions += rollouts * horizon
        # Each public rollout exposes reset, post-acquisition, and post-control
        # observations: 2*horizon + 1 ambient vectors.
        scalar_elements += rollouts * (2 * horizon + 1) * width
    return {
        "decision_invocations": sum(int(row["horizon"]) for row in rows),
        "threshold_table_lookups": 0,
        "scalar_threshold_comparisons": 0,
        "public_rollouts": sum(int(row["rollout_count"]) for row in rows),
        "simulated_control_transitions": transitions,
        "ambient_observation_scalar_elements": scalar_elements,
        "candidate_branches": sum(int(row["ambient_branch_count"]) for row in rows),
    }


def _representative_timing(
    *,
    pair: tuple[Mapping[str, Any], Mapping[str, Any]],
    confirmation: ConditionalEfficiencyConfirmationConfig,
    gate_model: GateModel,
    config: ComputeCharacterizationConfig,
) -> dict[str, Any]:
    gate_row, _ = pair
    family = _family_for_row(gate_row, confirmation)
    observation = SequentialQuotientSensingEnv(
        family, seed=int(gate_row["seed"])
    ).reset(seed=int(gate_row["seed"]))
    features = {
        **_controller_features(family),
        "probe_available": bool(observation["probe_available"]),
    }
    seed = int(gate_row["seed"])

    def rollout(*, rollout_seed: int, acquire: bool) -> dict[str, Any]:
        return _public_rollout(family, seed=rollout_seed, acquire=acquire)

    adapters: dict[str, InformationPolicy] = {
        "learned_causal_quotient": GatePolicyAdapter(
            "learned_causal_quotient", gate_model
        ),
        "generic_bounded_rollout_voi": BoundedRolloutVoiPolicyAdapter(
            budget=config.primary_rollout_budget
        ),
    }
    timings: dict[str, Any] = {}
    for name, adapter in adapters.items():
        batch_size = config.gate_batch_size if name == "learned_causal_quotient" else 1
        for _ in range(config.warmup_repetitions):
            for _ in range(batch_size):
                adapter.decide(
                    features=features,
                    seed=seed,
                    rollout=rollout,
                    probe_already_acquired=False,
                )
        samples: list[float] = []
        for _ in range(config.measured_repetitions):
            started = time.perf_counter_ns()
            for _ in range(batch_size):
                adapter.decide(
                    features=features,
                    seed=seed,
                    rollout=rollout,
                    probe_already_acquired=False,
                )
            samples.append((time.perf_counter_ns() - started) / batch_size)
        timings[name] = {
            "elapsed_time_ns_per_decision_median": int(median(samples)),
            "elapsed_time_ns_per_decision_mean": round(fmean(samples), 3),
            "batch_size": batch_size,
            "samples_ns_per_decision": [round(sample, 3) for sample in samples],
        }
    return {
        "protocol": "one fixed positive held-out decision per arm; fixed-host elapsed time; warm-up then repeated measurements",
        "warmup_repetitions": config.warmup_repetitions,
        "measured_repetitions": config.measured_repetitions,
        "representative": {
            key: gate_row[key]
            for key in (
                "topology_instance_id",
                "distractor_count",
                "horizon",
                "probe_cost",
                "seed",
            )
        },
        "arms": timings,
    }


def _family_for_row(
    row: Mapping[str, Any], config: ConditionalEfficiencyConfirmationConfig
) -> Any:
    """Regenerate exactly one named fixture rather than replaying the matrix."""
    for index, spec in enumerate(build_confirmation_specs(config)):
        if (
            spec.regime == row["regime"]
            and spec.topology_instance_id == row["topology_instance_id"]
            and spec.distractor_count == int(row["distractor_count"])
            and spec.horizon == int(row["horizon"])
            and spec.probe_cost == float(row["probe_cost"])
        ):
            family = generate_causal_quotient_family(spec=spec, index=index)
            if family.family_id != row["family_id"]:
                raise ValueError("representative fixture does not match the locked row")
            return family
    raise ValueError("could not locate representative fixture in locked manifest")


__all__ = ["ComputeCharacterizationConfig", "characterize_locked_confirmation"]
