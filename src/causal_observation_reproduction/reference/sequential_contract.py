"""Validity contract for the sequential POSCM confirmation benchmark.

This is deliberately a small executable contract, not a second experiment
runner.  It makes the three preconditions of the planned comparison explicit:
paired arms see the same latent world, refinement can have delayed net value,
and every policy-visible feedback channel is accounted for.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

INFORMATION_ARMS = (
    "never_refine",
    "always_refine",
    "random_budget_matched",
    "uncertainty",
    "information_gain",
    "bounded_generic_voi",
    "capacity_matched_no_quotient",
    "causal_quotient",
    "oracle",
)
# The contract retains the concise causal-quotient label while the runner
# records the learned implementation explicitly.  Keeping this mapping here
# prevents configuration, runtime rows, and evidence artifacts from drifting.
RUNNER_POLICY_BY_CONTRACT_ARM = {
    "never_refine": "never_refine",
    "always_refine": "always_refine",
    "random_budget_matched": "random_budget_matched",
    "uncertainty": "uncertainty",
    "information_gain": "information_gain",
    "bounded_generic_voi": "generic_bounded_rollout_voi",
    "capacity_matched_no_quotient": "capacity_matched_no_quotient",
    "causal_quotient": "learned_causal_quotient",
    "oracle": "oracle",
}
RUNNER_POLICY_ARMS = tuple(
    RUNNER_POLICY_BY_CONTRACT_ARM[arm] for arm in INFORMATION_ARMS
)
POLICY_VISIBLE_CHANNELS = (
    "telemetry",
    "reward",
    "termination",
    "action_result",
    "message",
    "visual",
    "refinement_receipt",
)


def validate_smoke_manifest(manifest: Mapping[str, object]) -> None:
    """Validate the static contract that defines the smoke comparison.

    Keeping this check next to the executable episode contract prevents a
    configuration edit from silently weakening a validity gate.
    """
    if (
        manifest.get("schema_version")
        != "causal_observation_reproduction.sequential_poscm_manifest.v1"
    ):
        raise ValueError("unexpected sequential POSCM manifest schema")
    information_arms = manifest.get("information_arms")
    if not isinstance(information_arms, (list, tuple)):
        raise ValueError("manifest information arms must be a sequence")
    if tuple(str(arm) for arm in information_arms) != INFORMATION_ARMS:
        raise ValueError(
            "manifest information arms must match the paired-world contract"
        )

    episode = manifest.get("episode")
    if not isinstance(episode, Mapping):
        raise ValueError("manifest episode section is required")
    if episode.get("horizon", 0) < 2:
        raise ValueError(
            "delayed continuation value requires a horizon of at least two"
        )
    if tuple(episode.get("information_actions", ())) != ("none", "refine", "probe"):
        raise ValueError("manifest must declare the smoke information actions")
    if episode.get("immediate_information_reward") != 0.0:
        raise ValueError("smoke contract requires zero immediate information reward")

    gates = manifest.get("validity_gates")
    if not isinstance(gates, Mapping):
        raise ValueError("manifest validity gates are required")
    required_gates = (
        "paired_latent_worlds",
        "delayed_continuation_value",
        "complete_feedback_boundary",
        "outcome_blind_regime_assignment",
    )
    missing = [gate for gate in required_gates if gates.get(gate) != "required"]
    if missing:
        raise ValueError(f"manifest weakens required validity gates: {missing}")


@dataclass(frozen=True)
class FeedbackBoundary:
    telemetry: str
    reward: float
    termination: bool
    action_result: str
    message: str
    visual: str
    refinement_receipt: str

    def project(
        self, channels: tuple[str, ...] = POLICY_VISIBLE_CHANNELS
    ) -> tuple[object, ...]:
        unknown = set(channels) - set(POLICY_VISIBLE_CHANNELS)
        if unknown:
            raise ValueError(f"unknown feedback channels: {sorted(unknown)}")
        return tuple(getattr(self, name) for name in channels)


@dataclass(frozen=True)
class DelayedValueEpisode:
    """Two-stage episode whose information value occurs only at the later action."""

    latent_world_id: str
    hidden_state: int
    immediate_information_reward: float = 0.0
    later_correct_action_reward: float = 1.0
    acquisition_cost: float = 0.10

    def coarse_value(self) -> float:
        """A coarse policy must choose one of two equally likely later actions."""
        return 0.5 * self.later_correct_action_reward

    def refine_value(self) -> float:
        return (
            self.immediate_information_reward
            + self.later_correct_action_reward
            - self.acquisition_cost
        )

    def delayed_net_gain(self) -> float:
        return self.refine_value() - self.coarse_value()

    def validates_delayed_value(self) -> bool:
        return (
            self.immediate_information_reward == 0.0 and self.delayed_net_gain() > 0.0
        )


@dataclass(frozen=True)
class PairedWorldBlock:
    block_id: str
    episode: DelayedValueEpisode
    arms: tuple[str, ...] = INFORMATION_ARMS

    def __post_init__(self) -> None:
        if not self.block_id:
            raise ValueError("block_id is required")
        if not self.arms or len(set(self.arms)) != len(self.arms):
            raise ValueError("arms must be nonempty and unique")
        unknown = set(self.arms) - set(INFORMATION_ARMS)
        if unknown:
            raise ValueError(f"unsupported information arms: {sorted(unknown)}")

    def latent_world_ids_by_arm(self) -> dict[str, str]:
        return dict.fromkeys(self.arms, self.episode.latent_world_id)

    def has_matched_latent_worlds(self) -> bool:
        return len(set(self.latent_world_ids_by_arm().values())) == 1


def build_smoke_block(seed: int = 7) -> PairedWorldBlock:
    hidden_state = int(seed) % 2
    episode = DelayedValueEpisode(
        latent_world_id=f"sequential-smoke-{int(seed):04d}",
        hidden_state=hidden_state,
    )
    return PairedWorldBlock(
        block_id=f"paired-sequential-{int(seed):04d}", episode=episode
    )


def assert_complete_feedback_boundary(
    left: FeedbackBoundary,
    right: FeedbackBoundary,
    *,
    channels: tuple[str, ...] = POLICY_VISIBLE_CHANNELS,
) -> None:
    """Reject a declared alias if any admitted channel separates the transitions."""
    if left.telemetry == right.telemetry and left.project(channels) != right.project(
        channels
    ):
        raise ValueError(
            "telemetry alias is invalid: an admitted feedback channel separates the states"
        )


def smoke_contract(seed: int = 7) -> dict[str, object]:
    block = build_smoke_block(seed)
    left = FeedbackBoundary("coarse", 0.0, False, "ok", "", "frame-a", "none")
    right = FeedbackBoundary("coarse", 1.0, False, "ok", "", "frame-a", "none")
    leakage_detected = False
    try:
        assert_complete_feedback_boundary(left, right)
    except ValueError:
        leakage_detected = True
    return {
        "schema_version": "causal_observation_reproduction.sequential_poscm_contract.v1",
        "block_id": block.block_id,
        "latent_world_id": block.episode.latent_world_id,
        "matched_latent_worlds": block.has_matched_latent_worlds(),
        "immediate_information_reward": block.episode.immediate_information_reward,
        "delayed_net_gain": block.episode.delayed_net_gain(),
        "delayed_value_supported": block.episode.validates_delayed_value(),
        "feedback_leakage_detected": leakage_detected,
        "arms": list(block.arms),
    }
