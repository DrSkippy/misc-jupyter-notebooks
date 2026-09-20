# misc-jupyter-notebooks

An assortment of Jupyter notebooks with explorations, prototypes, demonstrations and doodles — mostly probability, simulation, and statistics, plus a few puzzles and small tools.

## Standalone notebooks

| Notebook | What it explores |
|---|---|
| `birthdays-coincidence-paradox.ipynb` | Birthday paradox: probability that two or more people in a group share a birthday, as a function of group size |
| `employment-stopping-problem.ipynb` | The hiring / secretary problem (optimal stopping for sequential sampling) |
| `checkout-line-simulations.ipynb` | Discrete-event simulation comparing checkout-line strategies (customers, arrival rates, service times, wait statistics) |
| `monte-carlo-experiments.ipynb` | Monte Carlo sampling with `scipy.stats` distributions, using the Titanic dataset |
| `population-sampling.ipynb` | Sampling experiments showing 1/√n scaling |
| `gamma-distribution.ipynb` | Gamma distribution and double-exponential curves |
| `poisson-change-points-detection.ipynb` | Detecting a single change point in a Poisson process (2014) |
| `tcp-congestion-control.ipynb` | Notes and simulation from a 2020 talk by Andrew Jenkins on TCP congestion control |
| `connected-puzzle-pieces.ipynb` | Random puzzle-piece placement and the growth of connected components |
| `simple-sudoku-solver.ipynb` | Constraint-elimination Sudoku solver |
| `coffee-grind-adjustments.ipynb` | Rules for adjusting espresso grinder settings from dose, time and temperature readings |

`card.data.rda` is an R data file kept alongside the notebooks.

## Projects

Larger efforts live in their own subdirectories.

### `CrashCourse-Causality/`
Notebooks (I–III) and data projects (P1–P3) working through a crash course in causal inference.

### `LaneQueueSimulations/`
A small Python package (`lane_queues`) for simulating queueing across lanes, with notebooks (`sim.ipynb`, `sim2.ipynb`) and tests. Managed with Poetry.

```bash
cd LaneQueueSimulations
poetry install
poetry run pytest
```

### `time_to_traffic_metric_design/`
Simulation and analysis for designing an onboarding health metric based on days-to-first-traffic, using gamma modeling and Monte Carlo cohort simulation. Includes a CLI, tests and four notebooks meant to be run in order. See its [README](time_to_traffic_metric_design/README.md).

```bash
cd time_to_traffic_metric_design
poetry install
poetry run pytest test/
```

## Running the notebooks

The standalone notebooks have no shared environment file. Typical dependencies are `numpy`, `scipy`, `pandas`, `matplotlib` and `seaborn`:

```bash
pip install jupyter numpy scipy pandas matplotlib seaborn
jupyter notebook
```

## License

MIT — see [LICENSE](LICENSE).
