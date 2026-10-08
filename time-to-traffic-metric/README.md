# Time-to-First-Traffic Metric Design

Statistical simulation and analysis for designing a reliable onboarding health metric based on
days-to-first-traffic. Uses gamma distribution modeling, Monte Carlo cohort simulation, and a
rolling business metrics time series to evaluate candidate metrics under realistic conditions.

## Background

When entities onboard to a platform, their days-to-first-traffic follow a right-skewed
distribution: most entities are fast, but a long tail (and a small never-traffic fraction)
distort the population mean. This project asks: **which summary statistic should we track,
and how should we interpret it over time?**

Key findings:

- The **median** is a more stable and less biased estimator than the mean under realistic
  mixture distributions.
- The **cohort health score M = mean(cohort_median / days_i)** detects distributional shape
  changes (e.g., bimodal cohorts) that raw mean days miss.
- Cumulative avg/median days start artificially low due to survivorship bias and only
  converge to population values after ~2–3× the median onboarding time.
- The **score > 1 ratio** (restricted to a fixed reference cohort) provides a [0, 1]-bounded
  leading indicator of whether recent onboarders are ahead of the historical median.

## Project Structure

```
.
├── bin/
│   └── run_simulation.py       # CLI: runs all scenarios, writes CSVs + scorecard
├── notebooks/
│   ├── 01-gamma-distribution.ipynb       # Fit and visualize the gamma model
│   ├── 02-monte-carlo-comparison.ipynb   # Mean vs median stability analysis
│   ├── 03-cohort-metric-analysis.ipynb   # Cohort health score M across scenarios
│   └── 04-monthly-business-metrics.ipynb # Rolling time-series simulation
├── output/                     # Generated plots and CSVs (not committed)
├── tests/                      # pytest suite
├── time_to_traffic_sim/
│   ├── business_metrics.py     # Growth curves and the monthly metrics time series (NB04)
│   ├── config.py               # Project paths and config.yaml loading
│   ├── gamma_params.py         # Gamma parameter fitting and distribution stats
│   ├── metrics.py              # Cohort health score M and bootstrap CI
│   └── simulation.py           # Monte Carlo cohort simulation engine (numpy, optional CuPy GPU backend)
├── config.yaml                 # All tunable parameters (single source of truth)
└── pyproject.toml
```

## Running

Everything runs on the stock `quay.io/jupyter/datascience-notebook` image without installing anything: the
notebooks and the CLI put the project folder on `sys.path` themselves, and every path is resolved from the
project folder, so they work from any working directory.

Run the notebooks in order (01 → 04); each builds on concepts established in the previous one. Plots are saved to
`output/`.

The CLI runs all scenario × cohort-size combinations, writes CSVs to `output/`, and prints a scorecard:

```bash
python bin/run_simulation.py
python bin/run_simulation.py --config path/to/config.yaml
```

Tests (pytest is not in the Jupyter image):

```bash
pip install pytest pytest-cov
python -m pytest --cov=time_to_traffic_sim --cov-report=term-missing
```

Outside Docker, `pip install -e ".[dev]"` installs the package with its dependencies and dev tools.

### GPU backend

`simulation.backend` in `config.yaml` selects the array library for the cohort Monte Carlo: `numpy` (default),
`cupy` (NVIDIA GPU, `pip install "cupy-cuda12x[ctk]"`) or `auto` (GPU when one is usable, else numpy). At the
default sizes (5,000 cohorts of at most 250) a whole notebook's simulations take well under a second on the CPU,
so the GPU only pays off for much larger runs, roughly `n_simulations × cohort_size` above 10^7. GPU draws come
from CuPy's generator, so results are reproducible per backend but not identical across backends.

## Configuration (`config.yaml`)

All parameters are centralized in `config.yaml`. Notebooks and the CLI load from it at startup.

### `gamma` — Distribution shape

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode_days` | `7.0` | Most common days-to-first-traffic (peak of PDF) |
| `median_days` | `45.0` | Population median days-to-first-traffic |

Shape k and scale θ are solved numerically from mode and median at runtime.

### `simulation` — Monte Carlo settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cohort_sizes` | `[20, 50, 100, 250]` | Cohort sizes to simulate (NB02, NB03) |
| `n_simulations` | `5000` | Independent cohort draws per size |
| `random_seed` | `42` | Seed for reproducibility |
| `never_traffic_fraction` | `0.01` | Fraction of entities who never pass traffic |
| `never_traffic_days` | `400.0` | Days assigned to never-traffic entities |
| `backend` | `numpy` | `numpy`, `cupy` or `auto`; see GPU backend above |

### `scenarios` — Cohort scenarios (NB03 and CLI)

| Scenario | Description |
|----------|-------------|
| `baseline` | θ × 1.0 — standard population |
| `high_performing` | θ × 0.5 — twice as fast |
| `low_performing` | θ × 2.0 — twice as slow |
| `mixed` | Bimodal: 70% fast (θ × 0.4) + 30% slow (θ × 3.0) |

### `growth` — Entity acquisition model (NB04)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `"exponential"` | `"linear"` or `"exponential"` |
| `n_months` | `96` | Number of months to simulate |
| `days_per_month` | `30` | Calendar days per month |
| `variation` | `0.10` | ±noise fraction applied to deterministic rate |
| `linear.start_rate` | `20` | Starting entities/month (linear only) |
| `linear.end_rate` | `60` | Ending entities/month (linear only) |
| `exponential.initial_rate` | `1000` | Entities/month at t=0 (exponential only) |
| `exponential.monthly_rate` | `-0.05` | Monthly growth rate, e.g. -0.05 = 5% monthly decline |

## Notebooks

### 01 — Gamma Distribution

Fits the gamma distribution to the configured mode and median. Visualizes the PDF (with mode,
median, and mean annotated) and the CDF (cumulative conversion benchmarks by day). Introduces
the never-traffic mixture distribution and shows how it inflates the mean while leaving the
median nearly unchanged.

### 02 — Mean vs Median as Cohort Estimators

Monte Carlo comparison: for each cohort size, how well does the sample mean and sample median
track the true population value across 5 000 simulated cohorts? Reports Coefficient of
Variation and accuracy-within-±20% for both estimators, under the pure gamma and realistic
mixture distributions. Conclusion: median is more stable at small N.

### 03 — Cohort Health Score M

Introduces M = mean(cohort_median / days_i), the per-cohort health score. Compares M across
four scenarios (baseline, high-performing, low-performing, bimodal) and shows that M detects
distributional shape differences — particularly bimodal cohorts — that raw mean days miss.
Includes a full scorecard and violin/KDE visualizations.

### 04 — Monthly Business Metrics Time Series

End-to-end rolling simulation over `n_months`. Tracks 15 monthly metrics:

| Column | Description |
|--------|-------------|
| `new_entities` | New entities joining that month |
| `n_entities` | Cumulative entities started |
| `n_no_traffic` | Started but not yet converted |
| `avg_days` | Cumulative mean days-to-traffic (converged entities) |
| `median_days` | Cumulative median days-to-traffic |
| `avg_score` | Mean M score across all converted entities |
| `n_laggards` | Count with score < 1 |
| `n_score_gt1` | Count with score > 1 |
| `n_ref_cohort` | Size of reference cohort (started ~median days ago) |
| `n_ref_gt1` | Ref-cohort entities with score > 1 |
| `score_gt1_ratio` | n_ref_gt1 / n_ref_cohort — bounded [0, 1] |
| `n_spot` | Entities achieving first traffic this month |
| `spot_mean` | Mean days-to-traffic of this month's new converters |
| `spot_median` | Median days-to-traffic of this month's new converters |
