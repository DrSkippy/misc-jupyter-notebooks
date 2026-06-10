"""Tests for cohort health metrics."""

import math

import numpy as np
import pytest

from time_to_traffic_sim.metrics import (
    bootstrap_confidence_interval,
    cohort_mean_normalized,
    normalize_by_cohort_median,
    summarize_metric_distribution,
)


def test_normalize_median_is_one() -> None:
    days = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    scores = normalize_by_cohort_median(days)
    # median of days is 30; score for the median entity should be 1.0
    assert abs(scores[2] - 1.0) < 1e-10


def test_normalize_fast_entity_above_one() -> None:
    days = np.array([5.0, 20.0, 40.0, 60.0, 100.0])
    scores = normalize_by_cohort_median(days)
    # fastest entity (5 days) → median/days > 1
    assert scores[0] > 1.0


def test_normalize_slow_entity_below_one() -> None:
    days = np.array([5.0, 20.0, 40.0, 60.0, 100.0])
    scores = normalize_by_cohort_median(days)
    # slowest entity (100 days) → median/days < 1
    assert scores[-1] < 1.0


def test_normalize_empty_array_returns_empty() -> None:
    result = normalize_by_cohort_median(np.array([]))
    assert result.size == 0


def test_cohort_mean_normalized_degenerate_is_one() -> None:
    # All entities at the same value → median/days_i = 1 for all, mean = 1
    days = np.full(100, 42.0)
    assert abs(cohort_mean_normalized(days) - 1.0) < 1e-10


def test_cohort_mean_normalized_above_one_with_variance() -> None:
    # Jensen's inequality: E[1/X] > 1/E[X] for any X > 0 with variance,
    # so mean(median/days_i) > 1 for any non-degenerate distribution
    rng = np.random.default_rng(0)
    days = rng.uniform(0.1, 100.0, size=10_000)
    assert cohort_mean_normalized(days) > 1.0


def test_cohort_mean_normalized_right_skewed_above_one() -> None:
    # Right-skewed gamma: mean > median → M > 1
    rng = np.random.default_rng(1)
    days = rng.gamma(shape=1.2, scale=50.0, size=10_000)
    score = cohort_mean_normalized(days)
    assert score > 1.05


def test_cohort_mean_normalized_empty_returns_nan() -> None:
    assert math.isnan(cohort_mean_normalized(np.array([])))


def test_bootstrap_ci_ordering(rng: np.random.Generator) -> None:
    days = rng.gamma(shape=1.2, scale=50.0, size=100)
    lower, upper = bootstrap_confidence_interval(days, n_boot=500, ci=0.95, rng=rng)
    assert lower < upper
    assert lower > 0.0


def test_bootstrap_ci_above_one_for_skewed(rng: np.random.Generator) -> None:
    # For right-skewed data, CI should sit above 1
    days = rng.gamma(shape=1.2, scale=50.0, size=200)
    lower, upper = bootstrap_confidence_interval(days, n_boot=500, ci=0.95, rng=rng)
    assert lower > 1.0


def test_summarize_metric_distribution_keys() -> None:
    rng = np.random.default_rng(2)
    scores = rng.beta(5, 2, size=1000) + 1.0  # shift above 1 to mimic real M values
    summary = summarize_metric_distribution(scores)
    for key in ("mean", "std", "cv", "p5", "p25", "median", "p75", "p95"):
        assert key in summary


def test_summarize_metric_distribution_ordering() -> None:
    rng = np.random.default_rng(3)
    scores = rng.gamma(shape=5, scale=0.3, size=1000) + 1.0
    s = summarize_metric_distribution(scores)
    assert s["p5"] < s["p25"] < s["median"] < s["p75"] < s["p95"]
