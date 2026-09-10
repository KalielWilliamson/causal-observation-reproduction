from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SummaryStats:
    mean: float
    median: float
    std: float
    iqr: float
    minimum: float
    maximum: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float

    def as_payload(self) -> dict[str, float]:
        return {key: float(value) for key, value in self.__dict__.items()}


def summarize_world_metric(
    values: list[float] | np.ndarray, *, bootstrap_repetitions: int, seed: int
) -> SummaryStats:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        data = np.asarray([0.0], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    means = []
    for _ in range(max(1, int(bootstrap_repetitions))):
        sample = data[rng.integers(0, data.size, size=data.size)]
        means.append(float(np.mean(sample)))
    return SummaryStats(
        mean=float(np.mean(data)),
        median=float(np.median(data)),
        std=float(np.std(data, ddof=0)),
        iqr=float(np.percentile(data, 75) - np.percentile(data, 25)),
        minimum=float(np.min(data)),
        maximum=float(np.max(data)),
        bootstrap_ci_low=float(np.percentile(means, 2.5)),
        bootstrap_ci_high=float(np.percentile(means, 97.5)),
    )


def paired_comparison(
    left: np.ndarray, right: np.ndarray, *, bootstrap_repetitions: int, seed: int
) -> dict[str, float | int]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    diff = right - left
    rng = np.random.default_rng(int(seed))
    boot = []
    for _ in range(max(1, int(bootstrap_repetitions))):
        sample = (
            diff[rng.integers(0, diff.size, size=diff.size)]
            if diff.size
            else np.asarray([0.0])
        )
        boot.append(float(np.mean(sample)))
    return {
        "wins": int(np.sum(left < right)),
        "ties": int(np.sum(left == right)),
        "losses": int(np.sum(left > right)),
        "mean_paired_difference": float(np.mean(diff)) if diff.size else 0.0,
        "median_paired_difference": float(np.median(diff)) if diff.size else 0.0,
        "bootstrap_ci_low": float(np.percentile(boot, 2.5)),
        "bootstrap_ci_high": float(np.percentile(boot, 97.5)),
    }


def clustered_noninferiority(
    differences: Sequence[float] | np.ndarray,
    clusters: Sequence[str] | np.ndarray,
    *,
    margin: float,
    bootstrap_repetitions: int,
    seed: int,
) -> dict[str, float | int | bool]:
    """Bootstrap a paired return difference at the independent cluster level.

    ``differences`` is candidate minus comparator, so non-inferiority holds
    exactly when the lower confidence bound is greater than ``-margin``.
    Episode rows within a sampled topology instance stay together; this avoids
    treating repeated seeds or factorial cells as independent graph draws.
    """
    values = np.asarray(differences, dtype=np.float64)
    labels = np.asarray(clusters, dtype=str)
    if values.size != labels.size:
        raise ValueError("differences and clusters must have equal length")
    if values.size == 0:
        raise ValueError(
            "clustered non-inferiority requires at least one paired difference"
        )
    if margin < 0.0:
        raise ValueError("non-inferiority margin must be nonnegative")
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, label in zip(values, labels, strict=True):
        grouped[str(label)].append(float(value))
    cluster_means = np.asarray(
        [np.mean(grouped[key]) for key in sorted(grouped)], dtype=np.float64
    )
    rng = np.random.default_rng(int(seed))
    samples = np.asarray(
        [
            float(
                np.mean(
                    cluster_means[
                        rng.integers(0, cluster_means.size, size=cluster_means.size)
                    ]
                )
            )
            for _ in range(max(1, int(bootstrap_repetitions)))
        ],
        dtype=np.float64,
    )
    lower = float(np.quantile(samples, 0.025))
    upper = float(np.quantile(samples, 0.975))
    estimate = float(np.mean(cluster_means))
    return {
        "cluster_count": int(cluster_means.size),
        "paired_episode_count": int(values.size),
        "mean_paired_difference": estimate,
        "noninferiority_margin": float(margin),
        "bootstrap_ci_low": lower,
        "bootstrap_ci_high": upper,
        "noninferior": bool(lower > -float(margin)),
    }
