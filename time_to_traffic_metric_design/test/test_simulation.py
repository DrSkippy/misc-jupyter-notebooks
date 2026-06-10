"""Tests for Monte Carlo cohort simulation."""

import numpy as np
import pandas as pd
import pytest

from time_to_traffic_sim.simulation import simulate_cohorts, simulate_mixed_cohort

EXPECTED_COLUMNS = {
    "sim_id",
    "cohort_mean",
    "cohort_median",
    "cohort_p25",
    "cohort_p75",
    "cohort_p90",
    "mean_score",
}


def test_output_shape(gamma_params: tuple[float, float], rng: np.random.Generator) -> None:
    k, theta = gamma_params
    df = simulate_cohorts(k, theta, cohort_size=50, n_simulations=100, rng=rng)
    assert df.shape == (100, 7)


def test_output_columns(gamma_params: tuple[float, float], rng: np.random.Generator) -> None:
    k, theta = gamma_params
    df = simulate_cohorts(k, theta, cohort_size=50, n_simulations=100, rng=rng)
    assert set(df.columns) == EXPECTED_COLUMNS


def test_no_nans(gamma_params: tuple[float, float], rng: np.random.Generator) -> None:
    k, theta = gamma_params
    df = simulate_cohorts(k, theta, cohort_size=50, n_simulations=200, rng=rng)
    assert not df.isnull().any().any()


def test_reproducibility(gamma_params: tuple[float, float]) -> None:
    k, theta = gamma_params
    df1 = simulate_cohorts(k, theta, cohort_size=50, n_simulations=100, rng=np.random.default_rng(7))
    df2 = simulate_cohorts(k, theta, cohort_size=50, n_simulations=100, rng=np.random.default_rng(7))
    pd.testing.assert_frame_equal(df1, df2)


def test_mean_score_positive(
    gamma_params: tuple[float, float], rng: np.random.Generator
) -> None:
    # mean(cohort_median / days_i) is always positive for gamma samples
    k, theta = gamma_params
    df = simulate_cohorts(k, theta, cohort_size=100, n_simulations=500, rng=rng)
    assert (df["mean_score"] > 0).all()


def test_never_traffic_entities_present(
    gamma_params: tuple[float, float],
) -> None:
    # With 5% never-traffic, ~5% of samples should equal never_traffic_days=400
    k, theta = gamma_params
    rng = np.random.default_rng(99)
    # Use large cohort * many sims to get a stable fraction estimate
    df = simulate_cohorts(
        k, theta, cohort_size=200, n_simulations=500,
        never_traffic_fraction=0.05, never_traffic_days=400.0,
        rng=rng,
    )
    # cohort_p90 should be pulled up toward 400 vs no-never-traffic baseline
    baseline = simulate_cohorts(k, theta, cohort_size=200, n_simulations=500, rng=np.random.default_rng(99))
    assert df["cohort_p90"].mean() > baseline["cohort_p90"].mean()


def test_never_traffic_raises_mean(gamma_params: tuple[float, float]) -> None:
    k, theta = gamma_params
    df_with = simulate_cohorts(
        k, theta, cohort_size=100, n_simulations=500,
        never_traffic_fraction=0.05, never_traffic_days=400.0,
        rng=np.random.default_rng(5),
    )
    df_without = simulate_cohorts(
        k, theta, cohort_size=100, n_simulations=500,
        rng=np.random.default_rng(5),
    )
    assert df_with["cohort_mean"].mean() > df_without["cohort_mean"].mean()


def test_never_traffic_zero_fraction_unchanged(gamma_params: tuple[float, float]) -> None:
    k, theta = gamma_params
    df = simulate_cohorts(
        k, theta, cohort_size=50, n_simulations=100,
        never_traffic_fraction=0.0, never_traffic_days=400.0,
        rng=np.random.default_rng(7),
    )
    df_baseline = simulate_cohorts(k, theta, cohort_size=50, n_simulations=100, rng=np.random.default_rng(7))
    pd.testing.assert_frame_equal(df, df_baseline)


def test_mixed_cohort_shape(gamma_params: tuple[float, float], rng: np.random.Generator) -> None:
    k, theta = gamma_params
    df = simulate_mixed_cohort(
        k, theta,
        cohort_size=50, n_simulations=100,
        fast_fraction=0.7, fast_theta_multiplier=0.4, slow_theta_multiplier=3.0,
        rng=rng,
    )
    assert df.shape == (100, 7)
    assert set(df.columns) == EXPECTED_COLUMNS
