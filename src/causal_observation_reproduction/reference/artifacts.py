from __future__ import annotations

import json
from typing import Any, Protocol, cast, runtime_checkable

from causal_observation_reproduction.reference.hashing import (
    stable_json_hash as stable_hash,
)


@runtime_checkable
class ArtifactPayload(Protocol):
    def as_payload(self) -> dict[str, Any]: ...


def payload_value(value: Any) -> Any:
    if isinstance(value, ArtifactPayload):
        return payload_value(value.as_payload())
    if isinstance(value, tuple):
        return [payload_value(item) for item in value]
    if isinstance(value, list):
        return [payload_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): payload_value(item) for key, item in value.items()}
    return value


def artifact_payload_hash(payload: ArtifactPayload | dict[str, Any]) -> str:
    return stable_hash(payload_value(payload))


def assert_json_roundtrip(payload: dict[str, Any]) -> dict[str, Any]:
    return cast(
        "dict[str, Any]",
        json.loads(json.dumps(payload_value(payload), sort_keys=True)),
    )
