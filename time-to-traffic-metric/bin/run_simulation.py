#!/usr/bin/env python3
"""Run every scenario × cohort-size simulation, write one CSV each plus a scorecard, and print the scorecard.

    python bin/run_simulation.py [--config path/to/config.yaml]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root, so the package imports
from time_to_traffic_sim.config import DEFAULT_CONFIG, load_config, output_dir
from time_to_traffic_sim.gamma_params import describe_distribution, fit_gamma_params
from time_to_traffic_sim.metrics import summarize_metric_distribution
from time_to_traffic_sim.simulation import run_scenario

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main(config_path: str) -> pd.DataFrame:
    cfg = load_config(config_path)
    out_dir = output_dir(cfg)
    sim = cfg["simulation"]

    k, theta = fit_gamma_params(cfg["gamma"]["mode_days"], cfg["gamma"]["median_days"])
    stats = describe_distribution(k, theta)
    log.info("Gamma fit: shape k=%.4f, scale θ=%.4f", k, theta)
    log.info("  mean=%.1f  std=%.1f  median=%.1f  p90=%.1f days",
             stats["mean"], stats["std"], stats["median"], stats["p90"])
    log.info("Never-traffic entities: %.0f%% assigned %.0f days",
             sim["never_traffic_fraction"] * 100, sim["never_traffic_days"])

    rows = []
    for cohort_size in sim["cohort_sizes"]:
        log.info("Cohort size %d …", cohort_size)
        for key, scenario in cfg["scenarios"].items():
            df = run_scenario(
                k, theta, scenario, cohort_size, sim["n_simulations"],
                never_traffic_fraction=sim["never_traffic_fraction"],
                never_traffic_days=sim["never_traffic_days"],
                rng=np.random.default_rng(sim["random_seed"]),
                backend=sim.get("backend", "numpy"),
            )
            df["scenario"] = scenario["label"]
            df["cohort_size"] = cohort_size
            df.to_csv(out_dir / f"{key}_n{cohort_size}.csv", index=False)

            summary = summarize_metric_distribution(df["mean_score"].to_numpy())
            rows.append({
                "scenario": scenario["label"],
                "cohort_size": cohort_size,
                "M_mean": summary["mean"],
                "M_std": summary["std"],
                "M_p5": summary["p5"],
                "M_median": summary["median"],
                "M_p95": summary["p95"],
                "mean_days_mean": df["cohort_mean"].mean(),
                "median_days_mean": df["cohort_median"].mean(),
            })

    scorecard = pd.DataFrame(rows)
    scorecard.to_csv(out_dir / "scorecard.csv", index=False)
    print("\n=== Cohort Health Score (M = mean(cohort_median/days_i)) Scorecard ===\n")
    print(scorecard.to_string(index=False, float_format="{:.3f}".format))
    print(f"\nFull results written to {out_dir}/")
    return scorecard


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="path to config.yaml")
    main(parser.parse_args().config)
