"""Solve for gamma distribution parameters matching a target mode and median."""

from scipy.optimize import brentq
from scipy.stats import gamma as gamma_dist


def fit_gamma_params(mode_days: float, median_days: float) -> tuple[float, float]:
    """Return (shape k, scale θ) for Gamma(k, θ) matching the given mode and median.

    Args:
        mode_days: Target mode of the distribution in days.
        median_days: Target median of the distribution in days.

    Returns:
        Tuple of (shape k, scale θ) suitable for scipy.stats.gamma(a=k, scale=θ).

    Raises:
        ValueError: If no solution exists in the search bracket.
    """
    if mode_days <= 0 or median_days <= 0:
        raise ValueError("mode_days and median_days must be positive")
    if median_days <= mode_days:
        raise ValueError("median_days must exceed mode_days for a right-skewed distribution")

    def residual(k: float) -> float:
        theta = mode_days / (k - 1.0)
        return float(gamma_dist.ppf(0.5, a=k, scale=theta)) - median_days

    # k must be > 1 for the mode to be finite and positive
    k_sol: float = brentq(residual, 1.001, 100.0, xtol=1e-8)
    theta_sol: float = mode_days / (k_sol - 1.0)
    return k_sol, theta_sol


def describe_distribution(k: float, theta: float) -> dict[str, float]:
    """Return summary statistics for Gamma(k, θ).

    Args:
        k: Shape parameter.
        theta: Scale parameter.

    Returns:
        Dict with keys: mean, std, mode, median, p10, p25, p75, p90, p95.
    """
    dist = gamma_dist(a=k, scale=theta)
    return {
        "mean": float(dist.mean()),
        "std": float(dist.std()),
        "mode": (k - 1.0) * theta,
        "median": float(dist.ppf(0.5)),
        "p10": float(dist.ppf(0.10)),
        "p25": float(dist.ppf(0.25)),
        "p75": float(dist.ppf(0.75)),
        "p90": float(dist.ppf(0.90)),
        "p95": float(dist.ppf(0.95)),
    }
