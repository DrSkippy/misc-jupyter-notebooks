"""Monte Carlo cohort simulation for time-to-first-traffic."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _mean_score(samples: np.ndarray, cohort_medians: np.ndarray) -> np.ndarray:
    """Compute mean(cohort_median / days_i) per simulation row.

    Returns NaN for any row where the cohort median is zero.
    samples shape: (n_simulations, cohort_size)
    cohort_medians shape: (n_simulations,)
    """
    safe = np.where(cohort_medians > 0, cohort_medians, np.nan)
    return (safe[:, np.newaxis] / samples).mean(axis=1)


def _apply_never_traffic(
    samples: np.ndarray,
    never_traffic_fraction: float,
    never_traffic_days: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Replace a random fraction of entity samples with never_traffic_days.

    These entities represent those who never generate traffic within the
    observation window and are assigned the ceiling value.
    """
    if never_traffic_fraction <= 0.0:
        return samples
    mask = rng.random(size=samples.shape) < never_traffic_fraction
    return np.where(mask, never_traffic_days, samples)


def simulate_cohorts(
    k: float,
    theta: float,
    cohort_size: int,
    n_simulations: int,
    never_traffic_fraction: float = 0.0,
    never_traffic_days: float = 400.0,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Simulate many cohorts drawn from Gamma(k, θ) and record per-cohort statistics.

    A fraction `never_traffic_fraction` of entities in each cohort are assigned
    `never_traffic_days` to represent entities who never pass traffic.

    Args:
        k: Gamma shape parameter.
        theta: Gamma scale parameter.
        cohort_size: Number of entities per cohort.
        n_simulations: Number of independent cohort draws.
        never_traffic_fraction: Fraction of entities who never pass traffic.
        never_traffic_days: Days assigned to never-traffic entities.
        rng: Optional random generator for reproducibility.

    Returns:
        DataFrame with one row per simulation and columns:
        sim_id, cohort_mean, cohort_median, cohort_p25, cohort_p75, cohort_p90,
        mean_score (mean of per-entity cohort_median/days_i scores).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Draw all samples at once: shape (n_simulations, cohort_size)
    samples = rng.gamma(shape=k, scale=theta, size=(n_simulations, cohort_size))
    samples = _apply_never_traffic(samples, never_traffic_fraction, never_traffic_days, rng)

    cohort_means = samples.mean(axis=1)
    cohort_medians = np.median(samples, axis=1)
    cohort_p25 = np.percentile(samples, 25, axis=1)
    cohort_p75 = np.percentile(samples, 75, axis=1)
    cohort_p90 = np.percentile(samples, 90, axis=1)

    return pd.DataFrame(
        {
            "sim_id": np.arange(n_simulations),
            "cohort_mean": cohort_means,
            "cohort_median": cohort_medians,
            "cohort_p25": cohort_p25,
            "cohort_p75": cohort_p75,
            "cohort_p90": cohort_p90,
            "mean_score": _mean_score(samples, cohort_medians),
        }
    )


def simulate_mixed_cohort(
    k: float,
    theta: float,
    cohort_size: int,
    n_simulations: int,
    fast_fraction: float,
    fast_theta_multiplier: float,
    slow_theta_multiplier: float,
    never_traffic_fraction: float = 0.0,
    never_traffic_days: float = 400.0,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Simulate cohorts drawn from a bimodal mixture of two gamma distributions.

    The mixture has `fast_fraction` of entities drawn from Gamma(k, theta * fast_theta_multiplier)
    and the remainder from Gamma(k, theta * slow_theta_multiplier). An additional
    `never_traffic_fraction` of all entities are assigned `never_traffic_days`.

    Args:
        k: Gamma shape parameter (shared across both components).
        theta: Base gamma scale parameter.
        cohort_size: Number of entities per cohort.
        n_simulations: Number of independent cohort draws.
        fast_fraction: Fraction of entities in the fast component.
        fast_theta_multiplier: Scale multiplier for the fast component.
        slow_theta_multiplier: Scale multiplier for the slow component.
        never_traffic_fraction: Fraction of entities who never pass traffic.
        never_traffic_days: Days assigned to never-traffic entities.
        rng: Optional random generator for reproducibility.

    Returns:
        Same schema as simulate_cohorts (includes mean_score column).
    """
    if rng is None:
        rng = np.random.default_rng()

    n_fast = int(round(cohort_size * fast_fraction))
    n_slow = cohort_size - n_fast

    fast = rng.gamma(shape=k, scale=theta * fast_theta_multiplier, size=(n_simulations, n_fast))
    slow = rng.gamma(shape=k, scale=theta * slow_theta_multiplier, size=(n_simulations, n_slow))
    samples = np.concatenate([fast, slow], axis=1)
    samples = _apply_never_traffic(samples, never_traffic_fraction, never_traffic_days, rng)

    cohort_means = samples.mean(axis=1)
    cohort_medians = np.median(samples, axis=1)
    cohort_p25 = np.percentile(samples, 25, axis=1)
    cohort_p75 = np.percentile(samples, 75, axis=1)
    cohort_p90 = np.percentile(samples, 90, axis=1)

    return pd.DataFrame(
        {
            "sim_id": np.arange(n_simulations),
            "cohort_mean": cohort_means,
            "cohort_median": cohort_medians,
            "cohort_p25": cohort_p25,
            "cohort_p75": cohort_p75,
            "cohort_p90": cohort_p90,
            "mean_score": _mean_score(samples, cohort_medians),
        }
    )
