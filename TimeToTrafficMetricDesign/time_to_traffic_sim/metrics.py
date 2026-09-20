"""Cohort health metrics for time-to-first-traffic analysis."""

from __future__ import annotations

import numpy as np


def normalize_by_cohort_median(days: np.ndarray) -> np.ndarray:
    """Return per-entity scores: median(cohort) / days_i.

    Each score expresses an entity's speed relative to the cohort median.
    Scores > 1 are faster than median (took fewer days); scores < 1 are laggards.

    Args:
        days: Array of days-to-first-traffic for each entity in the cohort.

    Returns:
        Array of the same length as days, or empty array if days is empty.
    """
    if days.size == 0:
        return np.array([], dtype=float)
    cohort_median = float(np.median(days))
    if cohort_median == 0.0:
        return np.full_like(days, fill_value=float("nan"), dtype=float)
    return cohort_median / days


def cohort_mean_normalized(days: np.ndarray) -> float:
    """Return the mean of per-entity scores = mean(median(cohort) / days_i).

    Note: this is NOT equal to median/mean; it is the harmonic-mean-based ratio
    cohort_median * mean(1/days_i). Values > 1 when fast entities (small days)
    dominate; values approaching 0 when laggards dominate.

    Args:
        days: Array of days-to-first-traffic for each entity in the cohort.

    Returns:
        mean(median / days_i), or NaN if median is zero or array is empty.
    """
    if days.size == 0:
        return float("nan")
    cohort_median = float(np.median(days))
    if cohort_median == 0.0:
        return float("nan")
    return float(np.mean(cohort_median / days))


def bootstrap_confidence_interval(
    days: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Bootstrap percentile confidence interval for the cohort mean-normalized score.

    Args:
        days: Array of days-to-first-traffic values.
        n_boot: Number of bootstrap resamples.
        ci: Confidence level (e.g. 0.95 for 95% CI).
        rng: Optional random generator for reproducibility.

    Returns:
        Tuple (lower, upper) bounds of the confidence interval.
    """
    if rng is None:
        rng = np.random.default_rng()

    boot_scores = np.empty(n_boot)
    n = len(days)
    for i in range(n_boot):
        resample = rng.choice(days, size=n, replace=True)
        boot_scores[i] = cohort_mean_normalized(resample)

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_scores, 100.0 * alpha))
    upper = float(np.percentile(boot_scores, 100.0 * (1.0 - alpha)))
    return lower, upper


def summarize_metric_distribution(scores: np.ndarray) -> dict[str, float]:
    """Summarize a distribution of cohort-level M scores across many simulations.

    Args:
        scores: Array of M values (one per simulated cohort).

    Returns:
        Dict with keys: mean, std, cv, p5, p25, median, p75, p95.
    """
    mean = float(np.mean(scores))
    std = float(np.std(scores))
    return {
        "mean": mean,
        "std": std,
        "cv": std / mean if mean > 0 else float("nan"),
        "p5": float(np.percentile(scores, 5)),
        "p25": float(np.percentile(scores, 25)),
        "median": float(np.median(scores)),
        "p75": float(np.percentile(scores, 75)),
        "p95": float(np.percentile(scores, 95)),
    }
