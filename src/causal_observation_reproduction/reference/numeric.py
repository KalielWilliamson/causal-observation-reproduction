"""Small numeric coercion and aggregation helpers shared across runtime modules."""

from __future__ import annotations

from collections.abc import Iterable
from statistics import fmean, pvariance, variance
from typing import Any

TRUE_STRINGS = frozenset({"1", "true", "yes", "on"})
FALSE_STRINGS = frozenset({"0", "false", "no", "off"})


def coerce_int(value: Any, default: int = 0, *, strict: bool = False) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        if strict:
            raise ValueError(f"cannot coerce {value!r} to int") from exc
        return int(default)


def coerce_float(
    value: Any,
    default: float = 0.0,
    *,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> float:
    try:
        return float(value)
    except exceptions:
        return float(default)


def coerce_optional_float(
    value: Any,
    default: float | None = None,
    *,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> float | None:
    if value is None:
        return None if default is None else float(default)
    try:
        return float(value)
    except exceptions:
        return None if default is None else float(default)


def coerce_bool(value: Any, default: bool = False, *, strict: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in TRUE_STRINGS:
            return True
        if normalized in FALSE_STRINGS:
            return False
    if strict:
        raise ValueError(f"cannot coerce {value!r} to bool")
    return bool(default)


def clamp(value: Any, minimum: float = 0.0, maximum: float = 1.0) -> float:
    lower = float(minimum)
    upper = float(maximum)
    if lower > upper:
        lower, upper = upper, lower
    return max(lower, min(upper, float(value)))


def clamp01(value: Any) -> float:
    return clamp(value, 0.0, 1.0)


def safe_clamp01(value: Any, default: float = 0.0) -> float:
    return clamp01(coerce_float(value, default))


def _float_items(values: Iterable[Any]) -> list[float]:
    return [float(value) for value in values]


def mean(values: Iterable[Any], default: float = 0.0) -> float:
    items = _float_items(values)
    if not items:
        return float(default)
    return float(fmean(items))


def mean_or_none(values: Iterable[Any]) -> float | None:
    items = _float_items(values)
    if not items:
        return None
    return mean(items)


def population_variance(values: Iterable[Any], default: float = 0.0) -> float:
    items = _float_items(values)
    if not items:
        return float(default)
    return float(pvariance(items))


def sample_variance(values: Iterable[Any], default: float = 0.0) -> float:
    items = _float_items(values)
    if len(items) <= 1:
        return float(default)
    return float(variance(items))


__all__ = [
    "clamp",
    "clamp01",
    "coerce_bool",
    "coerce_float",
    "coerce_int",
    "coerce_optional_float",
    "mean",
    "mean_or_none",
    "population_variance",
    "safe_clamp01",
    "sample_variance",
]
