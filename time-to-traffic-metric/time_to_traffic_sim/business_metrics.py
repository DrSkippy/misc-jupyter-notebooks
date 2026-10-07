"""Rolling monthly business metrics for a growing (or shrinking) stream of onboarding entities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def growth_curve(growth: dict[str, Any]) -> tuple[np.ndarray, str]:
    """Deterministic entities-per-month curve from config.yaml's `growth` section.

    Returns (rates, label): one rate per month and a short description of the model.
    """
    n_months = growth["n_months"]
    if growth["model"] == "linear":
        lc = growth["linear"]
        return (np.linspace(lc["start_rate"], lc["end_rate"], n_months),
                f"linear {lc['start_rate']}→{lc['end_rate']} entities/mo")
    if growth["model"] == "exponential":
        ec = growth["exponential"]
        return (ec["initial_rate"] * (1 + ec["monthly_rate"]) ** np.arange(n_months),
                f"exponential {ec['initial_rate']} × (1+{ec['monthly_rate']})^t")
    raise ValueError(f"Unknown growth model: {growth['model']!r}. Use 'linear' or 'exponential'.")


def simulate_entities(
    k: float,
    theta: float,
    base_rates: np.ndarray,
    days_per_month: int,
    variation: float,
    never_traffic_fraction: float,
    never_traffic_days: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw every entity's start day and days-to-first-traffic.

    Each month's count is its base rate times uniform ±`variation` noise; start days are uniform within the month.
    Returns (monthly_counts, start_days sorted, days_to_traffic).
    """
    n_months = len(base_rates)
    monthly_counts = np.round(base_rates * rng.uniform(1 - variation, 1 + variation, size=n_months)).astype(int)
    start_days = np.sort(np.concatenate([
        rng.uniform(m * days_per_month + 1.0, (m + 1) * days_per_month, size=count)
        for m, count in enumerate(monthly_counts)
    ]))
    n = int(monthly_counts.sum())
    raw_days = rng.gamma(shape=k, scale=theta, size=n)
    never = rng.random(size=n) < never_traffic_fraction
    return monthly_counts, start_days, np.where(never, never_traffic_days, raw_days)


def monthly_metrics(
    monthly_counts: np.ndarray,
    start_days: np.ndarray,
    days_to_traffic: np.ndarray,
    days_per_month: int,
    lookback: float,
) -> pd.DataFrame:
    """Compute the business metrics at each month end.

    Columns (one row per month):
        new_entities     entities that started this month
        n_entities       entities started so far
        n_no_traffic     started but not yet converted
        avg_days         mean days-to-traffic of the converted entities
        median_days      median days-to-traffic of the converted entities
        avg_score        mean of median_days / days_i over the converted entities
        n_laggards       converted entities with score < 1
        n_score_gt1      converted entities with score > 1
        n_ref_cohort     entities that started in the month `lookback` days before this month end
        n_ref_gt1        converted reference-cohort entities with score > 1
        score_gt1_ratio  n_ref_gt1 / n_ref_cohort, in [0, 1]
        n_spot           entities whose first traffic falls in this month
        spot_mean        mean days-to-traffic of this month's new converters
        spot_median      median days-to-traffic of this month's new converters
    """
    traffic_day = start_days + days_to_traffic  # calendar day of first traffic
    records = []
    for month in range(1, len(monthly_counts) + 1):
        end_day = month * days_per_month

        started = start_days <= end_day
        converted = started & (end_day - start_days >= days_to_traffic)
        n_started, n_traffic = int(started.sum()), int(converted.sum())

        # Reference cohort: entities who started in the month containing (end_day - lookback)
        ref_day = end_day - lookback
        if ref_day >= 1:
            ref_m = int((ref_day - 1) // days_per_month)
            ref = (start_days >= ref_m * days_per_month + 1.0) & (start_days <= (ref_m + 1) * days_per_month)
        else:
            ref = np.zeros_like(started)
        n_ref_cohort = int(ref.sum())

        spot = days_to_traffic[(traffic_day > end_day - days_per_month) & (traffic_day <= end_day)]

        row = dict(month=month, new_entities=int(monthly_counts[month - 1]), n_entities=n_started,
                   n_no_traffic=n_started - n_traffic, avg_days=np.nan, median_days=np.nan, avg_score=np.nan,
                   n_laggards=0, n_score_gt1=0, n_ref_cohort=n_ref_cohort, n_ref_gt1=0)
        if n_traffic > 0:
            t_days = days_to_traffic[converted]
            median = float(np.median(t_days))
            scores = median / t_days
            row.update(avg_days=float(t_days.mean()), median_days=median, avg_score=float(scores.mean()),
                       n_laggards=int((scores < 1).sum()), n_score_gt1=int((scores > 1).sum()),
                       n_ref_gt1=int((median / days_to_traffic[ref & converted] > 1).sum()))
        row.update(score_gt1_ratio=row["n_ref_gt1"] / n_ref_cohort if n_ref_cohort else np.nan,
                   n_spot=len(spot),
                   spot_mean=float(spot.mean()) if len(spot) else np.nan,
                   spot_median=float(np.median(spot)) if len(spot) else np.nan)
        records.append(row)
    return pd.DataFrame(records)
