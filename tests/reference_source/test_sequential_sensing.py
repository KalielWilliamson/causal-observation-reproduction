from __future__ import annotations

import pytest

from causal_observation_reproduction.reference.quotient_family import (
    CausalQuotientFamilySpec,
    generate_causal_quotient_family,
)
from causal_observation_reproduction.reference.sequential_sensing import (
    SequentialQuotientSensingEnv,
)


def test_probe_is_budgeted_costed_and_absent_from_coarse_observation() -> None:
    family = generate_causal_quotient_family(
        spec=CausalQuotientFamilySpec(
            regime="positive",
            horizon=3,
            reward_delay=1,
            probe_cost=0.3,
            sensing_budget=1,
        )
    )
    env = SequentialQuotientSensingEnv(family, seed=7)
    coarse = env.reset(seed=7)
    assert coarse["probe_value"] is None
    assert "latent_context" not in coarse
    after_probe = env.acquire("probe_context")
    assert after_probe["budget_remaining"] == 0
    assert after_probe["probe_value"] in {0, 1}
    step = env.control(0)
    assert step.acquisition_cost == 0.3
    assert step.probe_attempted is True
    assert env.scorer_record()["deployment_visible"] is False


def test_information_must_precede_exactly_one_control_action() -> None:
    family = generate_causal_quotient_family(
        spec=CausalQuotientFamilySpec(regime="positive", horizon=3, reward_delay=1)
    )
    env = SequentialQuotientSensingEnv(family, seed=7)
    env.reset(seed=7)
    with pytest.raises(RuntimeError, match="information action"):
        env.control(0)
