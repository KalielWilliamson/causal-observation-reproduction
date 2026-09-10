from __future__ import annotations

from causal_observation_reproduction.reference.artifacts import (
    artifact_payload_hash,
    assert_json_roundtrip,
    payload_value,
)
from causal_observation_reproduction.reference.scm_complexity import StructuralTermSpec


def test_payload_value_normalizes_nested_artifact_payloads() -> None:
    term = StructuralTermSpec(
        term_type="interaction",
        coefficient=0.25,
        variables=("x1", "x2"),
        powers=(1, 1),
    )
    payload = payload_value({"terms": (term,), "edge": ("x1", "x2")})

    assert payload == {
        "terms": [
            {
                "term_type": "interaction",
                "coefficient": 0.25,
                "variables": ["x1", "x2"],
                "powers": [1, 1],
            }
        ],
        "edge": ["x1", "x2"],
    }
    assert assert_json_roundtrip(payload) == payload


def test_artifact_payload_hash_is_stable_across_tuple_list_equivalents() -> None:
    left = {"edge": ("x1", "x2")}
    right = {"edge": ["x1", "x2"]}

    assert artifact_payload_hash(left) == artifact_payload_hash(right)
