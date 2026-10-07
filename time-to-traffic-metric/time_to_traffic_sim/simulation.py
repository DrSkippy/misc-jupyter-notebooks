"""Monte Carlo cohort simulation for time-to-first-traffic.

All cohorts for one call are drawn as a single (n_simulations, cohort_size) array and summarized with vectorized
reductions. `backend="cupy"` runs the draws and reductions on an NVIDIA GPU (needs `cupy`); `"auto"` uses the GPU
when one is usable and numpy otherwise. On the GPU the draws come from CuPy's generator (seeded from `rng`), so
results are reproducible per backend but differ between backends. The GPU only pays off for large runs, roughly
n_simulations × cohort_size above 10^7; at the default config sizes numpy is faster.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd


def get_array_module(backend: str = "numpy") -> ModuleType:
    """Return numpy or cupy for backend 'numpy', 'cupy' or 'auto'."""
    if backend == "numpy":
        return np
    try:
        import cupy

        if cupy.cuda.runtime.getDeviceCount() > 0:
            return cupy  # type: ignore[no-any-return]
    except Exception:
        if backend == "cupy":
            raise
    if backend == "cupy":
        raise RuntimeError("backend='cupy' but no CUDA device is available")
    return np


def _generator(xp: ModuleType, rng: np.random.Generator) -> Any:
    """A random generator for the array module; GPU generators are seeded from `rng`."""
    return rng if xp is np else xp.random.default_rng(int(rng.integers(2**63)))


def _apply_never_traffic(
    xp: ModuleType, samples: Any, never_traffic_fraction: float, never_traffic_days: float, gen: Any
) -> Any:
    """Replace a random fraction of entity samples with never_traffic_days.

    These entities represent those who never generate traffic within the
    observation window and are assigned the ceiling value.
    """
    if never_traffic_fraction <= 0.0:
        return samples
    mask = gen.random(size=samples.shape) < never_traffic_fraction
    return xp.where(mask, never_traffic_days, samples)


def _summarize_cohorts(xp: ModuleType, samples: Any) -> pd.DataFrame:
    """One row per simulated cohort (row of `samples`): mean, quantiles and mean_score.

    mean_score is mean(cohort_median / days_i); NaN where the cohort median is zero.
    """
    medians = xp.median(samples, axis=1)
    p25, p75, p90 = xp.percentile(samples, [25, 75, 90], axis=1)
    safe = xp.where(medians > 0, medians, xp.nan)
    columns = {
        "cohort_mean": samples.mean(axis=1),
        "cohort_median": medians,
        "cohort_p25": p25,
        "cohort_p75": p75,
        "cohort_p90": p90,
        "mean_score": (safe[:, None] / samples).mean(axis=1),
    }
    to_numpy = (lambda a: a) if xp is np else xp.asnumpy
    df = pd.DataFrame({name: to_numpy(col) for name, col in columns.items()})
    df.insert(0, "sim_id", np.arange(len(df)))
    return df


def simulate_cohorts(
    k: float,
    theta: float,
    cohort_size: int,
    n_simulations: int,
    never_traffic_fraction: float = 0.0,
    never_traffic_days: float = 400.0,
    rng: np.random.Generator | None = None,
    backend: str = "numpy",
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
        backend: 'numpy', 'cupy' or 'auto' (see module docstring).

    Returns:
        DataFrame with one row per simulation and columns:
        sim_id, cohort_mean, cohort_median, cohort_p25, cohort_p75, cohort_p90,
        mean_score (mean of per-entity cohort_median/days_i scores).
    """
    xp = get_array_module(backend)
    gen = _generator(xp, rng if rng is not None else np.random.default_rng())
    samples = gen.gamma(shape=k, scale=theta, size=(n_simulations, cohort_size))
    samples = _apply_never_traffic(xp, samples, never_traffic_fraction, never_traffic_days, gen)
    return _summarize_cohorts(xp, samples)


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
    backend: str = "numpy",
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
        backend: 'numpy', 'cupy' or 'auto' (see module docstring).

    Returns:
        Same schema as simulate_cohorts (includes mean_score column).
    """
    xp = get_array_module(backend)
    gen = _generator(xp, rng if rng is not None else np.random.default_rng())

    n_fast = int(round(cohort_size * fast_fraction))
    n_slow = cohort_size - n_fast
    fast = gen.gamma(shape=k, scale=theta * fast_theta_multiplier, size=(n_simulations, n_fast))
    slow = gen.gamma(shape=k, scale=theta * slow_theta_multiplier, size=(n_simulations, n_slow))
    samples = xp.concatenate([fast, slow], axis=1)
    samples = _apply_never_traffic(xp, samples, never_traffic_fraction, never_traffic_days, gen)
    return _summarize_cohorts(xp, samples)


def run_scenario(
    k: float,
    theta: float,
    scenario: dict[str, Any],
    cohort_size: int,
    n_simulations: int,
    never_traffic_fraction: float = 0.0,
    never_traffic_days: float = 400.0,
    rng: np.random.Generator | None = None,
    backend: str = "numpy",
) -> pd.DataFrame:
    """Simulate one scenario from config.yaml's `scenarios` section.

    A scenario with `fast_fraction` is a bimodal mixture (simulate_mixed_cohort); otherwise its
    `theta_multiplier` scales θ (simulate_cohorts).
    """
    common = dict(
        cohort_size=cohort_size,
        n_simulations=n_simulations,
        never_traffic_fraction=never_traffic_fraction,
        never_traffic_days=never_traffic_days,
        rng=rng,
        backend=backend,
    )
    if "fast_fraction" in scenario:
        return simulate_mixed_cohort(
            k,
            theta,
            fast_fraction=scenario["fast_fraction"],
            fast_theta_multiplier=scenario["fast_theta_multiplier"],
            slow_theta_multiplier=scenario["slow_theta_multiplier"],
            **common,  # type: ignore[arg-type]
        )
    return simulate_cohorts(k, theta * scenario["theta_multiplier"], **common)  # type: ignore[arg-type]
