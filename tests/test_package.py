from __future__ import annotations

import importlib.metadata

import causal_observation_reproduction as m


def test_version() -> None:
    assert (
        importlib.metadata.version("causal_observation_reproduction") == m.__version__
    )
