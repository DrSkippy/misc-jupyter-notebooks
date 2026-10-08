"""Tests for the monthly business metrics simulation."""

import numpy as np
import pytest

from time_to_traffic_sim.business_metrics import growth_curve, monthly_metrics, simulate_entities


def test_growth_curve_models() -> None:
    linear, _ = growth_curve({"model": "linear", "n_months": 5, "linear": {"start_rate": 10, "end_rate": 50}})
    np.testing.assert_allclose(linear, [10, 20, 30, 40, 50])
    expo, _ = growth_curve({"model": "exponential", "n_months": 3,
                            "exponential": {"initial_rate": 100, "monthly_rate": 0.1}})
    np.testing.assert_allclose(expo, [100, 110, 121])
    with pytest.raises(ValueError):
        growth_curve({"model": "quadratic", "n_months": 3})


def test_monthly_metrics_hand_example() -> None:
    # Two entities start on days 1 and 31 and convert after 10 and 50 days (calendar days 11 and 81).
    counts = np.array([1, 1, 0])
    df = monthly_metrics(counts, np.array([1.0, 31.0]), np.array([10.0, 50.0]), days_per_month=30, lookback=20)
    assert df["n_entities"].tolist() == [1, 2, 2]
    assert df["n_no_traffic"].tolist() == [0, 1, 0]
    assert df["n_spot"].tolist() == [1, 0, 1]
    assert df.loc[2, "median_days"] == 30.0


def test_simulated_months_are_consistent(gamma_params: tuple[float, float]) -> None:
    k, theta = gamma_params
    rates = np.full(24, 200.0)
    counts, starts, days = simulate_entities(k, theta, rates, 30, 0.1, 0.01, 400.0, np.random.default_rng(0))
    df = monthly_metrics(counts, starts, days, days_per_month=30, lookback=45)
    assert df["n_entities"].iloc[-1] == counts.sum() == len(days)
    converted = df["n_entities"] - df["n_no_traffic"]
    assert (df["n_laggards"] + df["n_score_gt1"] <= converted).all()
    assert ((df["score_gt1_ratio"].dropna() >= 0) & (df["score_gt1_ratio"].dropna() <= 1)).all()
    assert df["n_spot"].sum() == (starts + days <= 24 * 30).sum()
