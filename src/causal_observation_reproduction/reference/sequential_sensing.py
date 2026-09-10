"""Budgeted, two-stage sensing wrapper for controlled quotient families.

Information acquisition happens before the control action at each decision
point.  The wrapper deliberately exposes only coarse state, budget, and an
optional noisy probe value; latent context and oracle quotient labels remain
internal scorer state.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Literal

from causal_observation_reproduction.reference.quotient_family import (
    GeneratedCausalQuotientFamily,
)
from causal_observation_reproduction.reference.scm_evaluator import (
    SemanticWorldScmEvaluator,
)

InformationAction = Literal["coarse", "probe_context"]


@dataclass(frozen=True)
class SensingStep:
    observation: dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    acquisition_cost: float
    probe_attempted: bool
    probe_succeeded: bool


class SequentialQuotientSensingEnv:
    """A finite-horizon environment with information/control action separation."""

    def __init__(self, family: GeneratedCausalQuotientFamily, *, seed: int = 0) -> None:
        self.family = family
        self.protocol = dict(family.deployment_manifest.get("protocol") or {})
        self.evaluator = SemanticWorldScmEvaluator(family.semantic_world, seed=seed)
        self.seed = int(seed)
        self._rng = random.Random(self.seed)
        self._context_variable = _context_variable(family.semantic_world)
        self._context_value = 0
        self._budget_remaining = 0
        self._probe_value: int | None = None
        self._probe_visible_until = -1
        self._pending_cost = 0.0
        self._pending_probe_attempted = False
        self._pending_probe_succeeded = False
        self._information_selected = False

    def reset(self, *, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.seed = int(seed)
        self._rng = random.Random(self.seed)
        self.evaluator.reset(seed=self.seed)
        self._context_value = int(self._rng.randrange(2))
        self.evaluator.state[self._context_variable] = float(self._context_value)
        self.evaluator.contextual_target_state = {
            self._context_variable: float(self._context_value)
        }
        self._budget_remaining = max(0, int(self.protocol.get("sensing_budget") or 0))
        self._probe_value = None
        self._probe_visible_until = -1
        self._pending_cost = 0.0
        self._pending_probe_attempted = False
        self._pending_probe_succeeded = False
        self._information_selected = False
        return self.observation()

    def observation(self) -> dict[str, Any]:
        visible_probe = (
            self._probe_value
            if self.evaluator.step_index <= self._probe_visible_until
            else None
        )
        return {
            "step_index": int(self.evaluator.step_index),
            "horizon": int(self.evaluator.horizon),
            "budget_remaining": int(self._budget_remaining),
            "probe_available": bool(self._budget_remaining > 0),
            "probe_value": visible_probe,
            # This is the full deployment-visible observation payload.  It is
            # intentionally separate from the scorer-only latent context so a
            # generic sensing policy can pay for its ambient dimensionality
            # without receiving the answer to the causal query for free.
            "ambient_observation": tuple(
                float(value) for value in self.evaluator.observation()
            ),
        }

    def acquire(self, action: InformationAction) -> dict[str, Any]:
        """Apply an information action without consuming a control time step."""
        if self._information_selected:
            raise RuntimeError(
                "exactly one information action is allowed before each control action"
            )
        self._information_selected = True
        if action not in {"coarse", "probe_context"}:
            raise ValueError("information action must be coarse or probe_context")
        if action == "probe_context" and self._budget_remaining > 0:
            self._budget_remaining -= 1
            self._pending_cost = float(self.protocol.get("probe_cost") or 0.0)
            reliability = float(self.protocol.get("probe_reliability") or 0.0)
            self._pending_probe_attempted = True
            self._pending_probe_succeeded = bool(self._rng.random() <= reliability)
            self._probe_value = (
                self._context_value
                if self._pending_probe_succeeded
                else 1 - self._context_value
            )
            persistence = max(1, int(self.protocol.get("alias_persistence") or 1))
            self._probe_visible_until = int(self.evaluator.step_index) + persistence - 1
        return self.observation()

    def control(self, action: int) -> SensingStep:
        """Execute an environment action after exactly one information action."""
        if not self._information_selected:
            raise RuntimeError(
                "select an information action before executing a control action"
            )
        observation, reward, terminated, truncated, _ = self.evaluator.step(int(action))
        del observation
        step = SensingStep(
            observation=self.observation(),
            reward=float(reward) - self._pending_cost,
            terminated=bool(terminated),
            truncated=bool(truncated),
            acquisition_cost=float(self._pending_cost),
            probe_attempted=bool(self._pending_probe_attempted),
            probe_succeeded=bool(self._pending_probe_succeeded),
        )
        self._pending_cost = 0.0
        self._pending_probe_attempted = False
        self._pending_probe_succeeded = False
        self._information_selected = False
        return step

    def scorer_record(self) -> dict[str, Any]:
        """Return hidden context only for scorer/audit code after an episode."""
        return {
            "artifact_scope": "scorer_only_oracle",
            "deployment_visible": False,
            "family_id": self.family.family_id,
            "latent_context": {self._context_variable: int(self._context_value)},
        }


def _context_variable(semantic_world: dict[str, Any]) -> str:
    rows = list(
        dict(semantic_world.get("reward_model") or {}).get(
            "context_required_action_sequences"
        )
        or []
    )
    if not rows:
        raise ValueError(
            "sequential sensing requires a controlled contextual action target"
        )
    context = dict(rows[0].get("context") or {})
    if len(context) != 1:
        raise ValueError(
            "controlled quotient family must declare exactly one context variable"
        )
    return str(next(iter(context)))


__all__ = ["InformationAction", "SensingStep", "SequentialQuotientSensingEnv"]
