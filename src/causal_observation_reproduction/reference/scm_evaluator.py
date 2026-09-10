"""SCM evaluator used by the manuscript sequential environment."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypedDict

import numpy as np
import sympy as sp

from causal_observation_reproduction.reference.semantic_world import (
    validate_semantic_world_payload,
)


class ContextActionSequence(TypedDict):
    context: dict[str, float]
    required_action_sequence: tuple[int, ...]


class SemanticWorldScmEvaluator:
    def __init__(
        self,
        semantic_world: dict[str, Any],
        *,
        seed: int | None = None,
        observation_mask: np.ndarray | list[float] | tuple[float, ...] | None = None,
    ) -> None:
        checks = validate_semantic_world_payload(semantic_world)
        if not all(checks.values()):
            failed = sorted(key for key, ok in checks.items() if not ok)
            raise ValueError(f"invalid semantic world for SCM evaluator: {failed}")
        self.semantic_world = dict(semantic_world)
        self.state_variables = [
            dict(row) for row in semantic_world.get("state_variables") or []
        ]
        self.transition_model = [
            dict(row) for row in semantic_world.get("transition_model") or []
        ]
        self.action_space = dict(semantic_world.get("action_space") or {})
        self.observation_model = dict(semantic_world.get("observation_model") or {})
        self.reward_model = dict(semantic_world.get("reward_model") or {})
        self.difficulty = dict(semantic_world.get("difficulty_vector") or {})
        self.variable_ids = [
            str(row.get("variable_id") or "") for row in self.state_variables
        ]
        self.observed_variables = [
            str(item) for item in self.observation_model.get("observed_variables") or []
        ]
        self.action_variables = [
            str(item) for item in self.action_space.get("action_variables") or []
        ]
        self.outcome_variables = [
            str(item) for item in self.reward_model.get("outcome_variables") or []
        ]
        self.observation_mask = _resolve_observation_mask(
            state_variables=self.state_variables,
            observation_mask=observation_mask,
        )
        self.compiled_transition_model = [
            _compile_transition_row(row) for row in self.transition_model
        ]
        self.horizon = max(1, int(float(self.difficulty.get("horizon") or 8.0)))
        self.reward_delay = max(0, int(self.reward_model.get("reward_delay") or 0))
        self.success_threshold = float(
            self.reward_model.get("success_threshold") or 0.5
        )
        self.required_action_sequence = tuple(
            int(action)
            for action in list(self.reward_model.get("required_action_sequence") or [])
        )
        context_action_sequences: list[ContextActionSequence] = []
        for raw_row in list(
            self.reward_model.get("context_required_action_sequences") or []
        ):
            if not isinstance(raw_row, Mapping):
                continue
            raw_context = raw_row.get("context")
            context = (
                {
                    str(variable_id): float(value)
                    for variable_id, value in raw_context.items()
                }
                if isinstance(raw_context, Mapping)
                else {}
            )
            raw_sequence = raw_row.get("required_action_sequence")
            sequence = (
                tuple(int(action) for action in raw_sequence)
                if isinstance(raw_sequence, (list, tuple))
                else ()
            )
            context_action_sequences.append(
                {"context": context, "required_action_sequence": sequence}
            )
        self.context_required_action_sequences = tuple(context_action_sequences)
        self.trap_action_sequences = tuple(
            tuple(int(action) for action in list(sequence))
            for sequence in list(self.reward_model.get("trap_action_sequences") or [])
        )
        trap_reward = self.reward_model.get(
            "trap_reward", self.reward_model.get("failure_reward") or 0.0
        )
        self.trap_reward = float(0.0 if trap_reward is None else trap_reward)
        self.trap_terminates = bool(self.reward_model.get("trap_terminates", True))
        self.action_effect_matrix = np.asarray(
            self.action_space.get("action_effect_matrix"), dtype=float
        )
        self.rng = np.random.default_rng(seed)
        self.state = dict.fromkeys(self.variable_ids, 0.0)
        self.contextual_target_state: dict[str, float] | None = None
        self.action_history: list[int] = []
        self.step_index = 0

    @property
    def observation_dim(self) -> int:
        return len(self.variable_ids) + 2

    @property
    def action_count(self) -> int:
        return max(
            1,
            int(
                self.action_space.get("action_count") or len(self.action_variables) or 1
            ),
        )

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_index = 0
        self.action_history = []
        self.state = {}
        for row in self.state_variables:
            variable_id = str(row.get("variable_id") or "")
            role = str(row.get("role") or "")
            self.state[variable_id] = float(
                self.rng.binomial(1, 0.25 if role in {"noise", "confounder"} else 0.0)
            )
        self.contextual_target_state = dict(self.state)
        return self.observation(), self.info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        selected_action = int(action) % self.action_count
        self.action_history.append(selected_action)
        action_values = self._action_values(action)
        next_state = dict(self.state)
        values = {**self.state, **action_values}
        for row in self.compiled_transition_model:
            variable_id = str(row["variable_id"])
            raw_value = _evaluate_compiled_transition(row, values)
            next_value = self._activation(raw_value + self._noise(variable_id))
            next_state[variable_id] = next_value
            values[variable_id] = next_value
        self.step_index += 1
        self.state = next_state
        trap_triggered = self.trap_triggered()
        success = self.success()
        terminated = bool(
            (trap_triggered and self.trap_terminates)
            or (success and self.step_index > self.reward_delay)
        )
        truncated = self.step_index >= self.horizon
        reward = self.reward(success=success, trap_triggered=trap_triggered)
        return (
            self.observation(),
            reward,
            terminated,
            truncated,
            self.info(action=action, reward=reward),
        )

    def observation(self) -> np.ndarray:
        noise = float(self.observation_model.get("observation_noise") or 0.0)
        observed = []
        for idx, variable_id in enumerate(self.variable_ids):
            value = float(self.state.get(variable_id, 0.0))
            mask_value = float(self.observation_mask[idx])
            if mask_value > 0.0 and noise > 0.0:
                value = float(
                    np.clip(value + self.rng.normal(0.0, noise * 0.25), 0.0, 1.0)
                )
            observed.append(float(mask_value * value))
        observed.append(float(self.step_index) / float(max(1, self.horizon)))
        observed.append(float(self.reward_delay) / float(max(1, self.horizon)))
        return np.asarray(observed, dtype=np.float32)

    def success(self) -> bool:
        if self.trap_triggered():
            return False
        contextual_sequence = self._context_required_action_sequence()
        if contextual_sequence is not None:
            return (
                len(self.action_history) >= len(contextual_sequence)
                and tuple(self.action_history[: len(contextual_sequence)])
                == contextual_sequence
            )
        if self.required_action_sequence:
            sequence = tuple(
                int(action) % self.action_count
                for action in self.required_action_sequence
            )
            return (
                len(self.action_history) >= len(sequence)
                and tuple(self.action_history[: len(sequence)]) == sequence
            )
        return any(
            float(self.state.get(variable_id, 0.0)) >= float(self.success_threshold)
            for variable_id in self.outcome_variables
        )

    def _context_required_action_sequence(self) -> tuple[int, ...] | None:
        """Resolve an optional controlled benchmark target from declared state.

        This extension is used by generated causal-quotient families.  It is
        not exposed in observations and only applies when a row matches the
        current state exactly on all of its declared context variables.
        """
        for row in self.context_required_action_sequences:
            context = dict(row["context"])
            source_state = self.contextual_target_state or self.state
            if context and all(
                float(source_state.get(key, float("nan"))) == value
                for key, value in context.items()
            ):
                return tuple(row["required_action_sequence"])
        return None

    def trap_triggered(self) -> bool:
        if not self.trap_action_sequences:
            return False
        history = tuple(
            int(action) % self.action_count for action in self.action_history
        )
        for sequence in self.trap_action_sequences:
            normalized = tuple(int(action) % self.action_count for action in sequence)
            if (
                normalized
                and len(history) >= len(normalized)
                and history[: len(normalized)] == normalized
            ):
                return True
        return False

    def reward(self, *, success: bool, trap_triggered: bool = False) -> float:
        if trap_triggered:
            return float(self.trap_reward)
        if self.step_index <= self.reward_delay:
            return 0.0
        if success:
            return float(self.reward_model.get("success_reward") or 1.0)
        return float(self.reward_model.get("failure_reward") or 0.0)

    def info(
        self, *, action: int | None = None, reward: float | None = None
    ) -> dict[str, Any]:
        return {
            "semantic_world_id": str(
                self.semantic_world.get("semantic_world_id") or ""
            ),
            "step_index": int(self.step_index),
            "horizon": int(self.horizon),
            "action": None if action is None else int(action),
            "reward": None if reward is None else float(reward),
            "success": bool(self.success()),
            "required_action_sequence": [
                int(action) for action in self.required_action_sequence
            ],
            "trap_action_sequences": [
                [int(action) for action in sequence]
                for sequence in self.trap_action_sequences
            ],
            "trap_reward": float(self.trap_reward),
            "trap_terminates": bool(self.trap_terminates),
            "trap_triggered": bool(self.trap_triggered()),
            "behavioural_failure": bool(self.trap_triggered()),
            "causal_advantage_signal": float(self.causal_advantage_signal()),
            "outcome_values": {
                variable_id: float(self.state.get(variable_id, 0.0))
                for variable_id in self.outcome_variables
            },
            "observed_variables": list(self.observed_variables),
        }

    def causal_advantage_signal(self) -> float:
        if self.trap_triggered():
            return -1.0
        if self.success():
            return 1.0
        if not self.required_action_sequence:
            return 0.0
        history = tuple(
            int(action) % self.action_count for action in self.action_history
        )
        required = tuple(
            int(action) % self.action_count for action in self.required_action_sequence
        )
        if not history:
            return 0.0
        prefix = required[: len(history)]
        return 0.25 if history == prefix else -0.25

    def greedy_action(self) -> int:
        scores = []
        for action in range(self.action_count):
            action_values = self._action_values(action)
            values = {**self.state, **action_values}
            score = 0.0
            for row in self.compiled_transition_model:
                variable_id = str(row["variable_id"])
                next_value = self._activation(
                    _evaluate_compiled_transition(row, values)
                )
                values[variable_id] = next_value
                if variable_id in set(self.outcome_variables):
                    score = max(score, next_value)
            scores.append(score)
        return int(max(range(len(scores)), key=lambda idx: (scores[idx], -idx)))

    def _action_values(self, action: int) -> dict[str, float]:
        selected = int(action) % self.action_count
        row = self.action_effect_matrix[selected]
        return {
            variable_id: float(row[idx])
            for idx, variable_id in enumerate(self.action_variables)
        }

    def _activation(self, value: float) -> float:
        return float(1.0 / (1.0 + np.exp(-float(value))))

    def _noise(self, variable_id: str) -> float:
        row = next(
            (
                item
                for item in self.state_variables
                if str(item.get("variable_id") or "") == variable_id
            ),
            {},
        )
        if str(row.get("role") or "") in {"noise", "confounder"}:
            return float(self.rng.normal(0.0, 0.15))
        return 0.0


def _resolve_observation_mask(
    *,
    state_variables: list[dict[str, Any]],
    observation_mask: np.ndarray | list[float] | tuple[float, ...] | None,
) -> np.ndarray:
    if observation_mask is None:
        return np.asarray(
            [1.0 if bool(row.get("observed")) else 0.0 for row in state_variables],
            dtype=np.float32,
        )
    mask = np.asarray(observation_mask, dtype=np.float32).reshape(-1)
    if mask.shape != (len(state_variables),):
        raise ValueError(
            "observation_mask must have one entry per semantic-world state variable "
            f"({len(state_variables)} expected, got {mask.shape[0]})"
        )
    return np.clip(mask, 0.0, 1.0)


def _compile_transition_row(row: dict[str, Any]) -> dict[str, Any]:
    structural_equation = dict(row.get("structural_equation") or {})
    inputs = [str(item) for item in structural_equation.get("inputs") or []]
    symbols = {name: sp.Symbol(name) for name in inputs}
    expression = sp.sympify(
        str(structural_equation.get("sympy_expression") or "0"), locals=symbols
    )
    fn = sp.lambdify([symbols[name] for name in inputs], expression, modules="numpy")
    return {
        "variable_id": str(row.get("variable_id") or ""),
        "inputs": inputs,
        "fn": fn,
    }


def _evaluate_compiled_transition(
    row: dict[str, Any], values: dict[str, float]
) -> float:
    result = row["fn"](*[float(values.get(name, 0.0)) for name in row["inputs"]])
    return float(np.asarray(result).reshape(-1)[0])
