"""Source-derived implementations for the manuscript experiments.

Only package/import names and public artifact metadata differ from the pinned
upstream implementation. Scientific behavior is protected by parity tests.
"""

from causal_observation_reproduction.reference.sequential_experiment import (
    ConditionalEfficiencyConfirmationConfig,
    ConditionalEfficiencyConfirmationResult,
    run_conditional_efficiency_confirmation,
)

__all__ = [
    "ConditionalEfficiencyConfirmationConfig",
    "ConditionalEfficiencyConfirmationResult",
    "run_conditional_efficiency_confirmation",
]
