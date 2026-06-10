"""Tests for gamma distribution parameter fitting."""

import pytest
from scipy.stats import gamma as gamma_dist

from time_to_traffic_sim.gamma_params import describe_distribution, fit_gamma_params


def test_fit_mode_and_median() -> None:
    k, theta = fit_gamma_params(mode_days=7.0, median_days=45.0)

    recovered_mode = (k - 1.0) * theta
    recovered_median = gamma_dist.ppf(0.5, a=k, scale=theta)

    assert abs(recovered_mode - 7.0) < 0.01, f"mode {recovered_mode:.4f} != 7.0"
    assert abs(recovered_median - 45.0) < 0.01, f"median {recovered_median:.4f} != 45.0"


def test_fit_requires_median_gt_mode() -> None:
    with pytest.raises(ValueError, match="median_days must exceed mode_days"):
        fit_gamma_params(mode_days=50.0, median_days=7.0)


def test_fit_requires_positive_inputs() -> None:
    with pytest.raises(ValueError):
        fit_gamma_params(mode_days=-1.0, median_days=45.0)


def test_describe_distribution_keys(gamma_params: tuple[float, float]) -> None:
    k, theta = gamma_params
    stats = describe_distribution(k, theta)
    for key in ("mean", "std", "mode", "median", "p10", "p25", "p75", "p90", "p95"):
        assert key in stats


def test_describe_distribution_ordering(gamma_params: tuple[float, float]) -> None:
    k, theta = gamma_params
    stats = describe_distribution(k, theta)
    assert stats["p10"] < stats["p25"] < stats["median"] < stats["mean"]
    assert stats["median"] < stats["p75"] < stats["p90"] < stats["p95"]
