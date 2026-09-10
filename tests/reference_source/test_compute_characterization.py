import pytest

pytest.importorskip("numpy")

from causal_observation_reproduction.reference.compute_characterization import (
    _gate_work,
    _voi_work,
)


def test_source_defined_work_counts_keep_gate_and_rollout_work_separate():
    gate = _gate_work(
        ({"probe_count": 1, "horizon": 3}, {"probe_count": 0, "horizon": 5})
    )
    voi = _voi_work(
        (
            {
                "distractor_count": 2,
                "rollout_count": 4,
                "horizon": 3,
                "ambient_branch_count": 2,
            },
        )
    )

    assert gate == {
        "decision_invocations": 8,
        "threshold_table_lookups": 6,
        "scalar_threshold_comparisons": 6,
        "public_rollouts": 0,
        "simulated_control_transitions": 0,
        "candidate_branches": 0,
    }
    assert voi["public_rollouts"] == 4
    assert voi["simulated_control_transitions"] == 12
    assert voi["ambient_observation_scalar_elements"] == 4 * (2 * 3 + 1) * (8 + 2)
    assert voi["candidate_branches"] == 2
