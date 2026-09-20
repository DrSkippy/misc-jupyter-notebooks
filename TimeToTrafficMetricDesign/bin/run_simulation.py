#!/usr/bin/env python3
"""CLI entry point for time-to-first-traffic Monte Carlo simulation."""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd
import yaml

from time_to_traffic_sim.gamma_params import describe_distribution, fit_gamma_params
from time_to_traffic_sim.metrics import summarize_metric_distribution
from time_to_traffic_sim.simulation import simulate_cohorts, simulate_mixed_cohort

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _load_config(path: str) -> dict:  # type: ignore[type-arg]
    with open(path) as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


@click.command()
@click.option("--config", default="config.yaml", show_default=True, help="Path to config.yaml")
def main(config: str) -> None:
    """Run Monte Carlo cohort simulations and write results to the output directory."""
    cfg = _load_config(config)
    out_dir = Path(cfg["output"]["directory"])
    out_dir.mkdir(exist_ok=True)

    mode_days: float = cfg["gamma"]["mode_days"]
    median_days: float = cfg["gamma"]["median_days"]

    log.info("Fitting gamma parameters: mode=%.1f days, median=%.1f days", mode_days, median_days)
    k, theta = fit_gamma_params(mode_days, median_days)
    stats = describe_distribution(k, theta)

    log.info("  shape k=%.4f, scale θ=%.4f", k, theta)
    log.info("  mean=%.1f  std=%.1f  median=%.1f  p90=%.1f days", stats["mean"], stats["std"], stats["median"], stats["p90"])

    seed: int = cfg["simulation"]["random_seed"]
    n_sims: int = cfg["simulation"]["n_simulations"]
    cohort_sizes: list[int] = cfg["simulation"]["cohort_sizes"]
    never_frac: float = cfg["simulation"]["never_traffic_fraction"]
    never_days: float = cfg["simulation"]["never_traffic_days"]
    scenarios_cfg: dict = cfg["scenarios"]

    log.info("Never-traffic entities: %.0f%% assigned %.0f days", never_frac * 100, never_days)

    all_rows: list[dict] = []

    for cohort_size in cohort_sizes:
        log.info("Cohort size %d …", cohort_size)
        for scenario_key, scenario_cfg in scenarios_cfg.items():
            label: str = scenario_cfg["label"]
            rng = np.random.default_rng(seed)

            if scenario_key == "mixed":
                df = simulate_mixed_cohort(
                    k, theta,
                    cohort_size=cohort_size, n_simulations=n_sims,
                    fast_fraction=scenario_cfg["fast_fraction"],
                    fast_theta_multiplier=scenario_cfg["fast_theta_multiplier"],
                    slow_theta_multiplier=scenario_cfg["slow_theta_multiplier"],
                    never_traffic_fraction=never_frac,
                    never_traffic_days=never_days,
                    rng=rng,
                )
            else:
                theta_m: float = scenario_cfg["theta_multiplier"]
                df = simulate_cohorts(
                    k, theta * theta_m,
                    cohort_size=cohort_size, n_simulations=n_sims,
                    never_traffic_fraction=never_frac,
                    never_traffic_days=never_days,
                    rng=rng,
                )

            df["scenario"] = label
            df["cohort_size"] = cohort_size
            out_path = out_dir / f"{scenario_key}_n{cohort_size}.csv"
            df.to_csv(out_path, index=False)

            summary = summarize_metric_distribution(df["mean_score"].to_numpy())
            all_rows.append({
                "scenario": label,
                "cohort_size": cohort_size,
                "M_mean": summary["mean"],
                "M_std": summary["std"],
                "M_p5": summary["p5"],
                "M_median": summary["median"],
                "M_p95": summary["p95"],
                "mean_days_mean": df["cohort_mean"].mean(),
                "median_days_mean": df["cohort_median"].mean(),
            })

    scorecard = pd.DataFrame(all_rows)
    scorecard_path = out_dir / "scorecard.csv"
    scorecard.to_csv(scorecard_path, index=False)

    click.echo("\n=== Cohort Health Score (M = mean(cohort_median/days_i)) Scorecard ===\n")
    click.echo(scorecard.to_string(index=False, float_format="{:.3f}".format))
    click.echo(f"\nFull results written to {out_dir}/")


if __name__ == "__main__":
    main()
