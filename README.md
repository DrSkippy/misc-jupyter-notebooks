# misc-jupyter-notebooks

An assortment of Jupyter notebooks with explorations, prototypes, demonstrations and doodles — mostly probability,
simulation, and statistics, plus a few puzzles and small tools.

Single-topic notebooks live at the top level. Anything with several notebooks or its own Python/R code has a
folder of its own.

```
.
├── *.ipynb                    # standalone notebooks (table below)
├── causality-crash-course/    # R: matching, IPW and instrumental variables (course notebooks)
├── checkout-queues/           # checkout-line queueing simulations (package + 3 notebooks + tests)
├── chip-bets/                 # 2017 betting contest: simulator, submissions, comparison notebook
├── time-to-traffic-metric/    # onboarding metric design (package + CLI + 4 notebooks + tests)
└── docker/                    # compose file for JupyterLab, optional Dockerfile with extras
```

## Standalone notebooks

| Notebook | What it explores |
|---|---|
| `birthdays-coincidence-paradox.ipynb` | Birthday paradox: probability that two (three, four) or more people in a group share a birthday, simulated and exact |
| `employment-stopping-problem.ipynb` | The hiring / secretary problem: optimal stopping for sequential sampling |
| `monte-carlo-experiments.ipynb` | `scipy.stats` distributions, fitting ages from the Titanic data, a Monte Carlo cost-savings risk estimate |
| `population-sampling.ipynb` | Sampling a synthetic population: 1/√n scaling, splitting on conditions, testing for a difference |
| `gamma-distribution.ipynb` | Gamma distribution vs a double-exponential rise-and-decay curve |
| `poisson-change-points-detection.ipynb` | Detecting a single change point in a Poisson process; the Skellam distribution (2014) |
| `tcp-congestion-control.ipynb` | Additive increase / multiplicative decrease, from a 2020 talk by Andrew Jenkins |
| `connected-puzzle-pieces.ipynb` | Random jigsaw piece placement and the growth of connections |
| `simple-sudoku-solver.ipynb` | Constraint-elimination Sudoku solver with backtracking |
| `coffee-grind-adjustments.ipynb` | Rules for adjusting espresso grinder settings from dose, time and temperature readings |

## Projects

| Folder | Contents |
|---|---|
| [`causality-crash-course/`](causality-crash-course/README.md) | R notebooks for *A Crash Course in Causality*: matching, IPW and MSMs, instrumental variables, and three data projects on the Lalonde data |
| [`checkout-queues/`](checkout-queues/README.md) | Power of d choices (join the shortest of d lines) and one shared line vs a line per clerk |
| [`chip-bets/`](chip-bets/README.md) | Guess-the-bag betting contest: the simulator, the submitted strategies, two AI strategies and a many-seed comparison |
| [`time-to-traffic-metric/`](time-to-traffic-metric/README.md) | Designing an onboarding health metric from days-to-first-traffic: gamma modeling, Monte Carlo cohorts, monthly metrics |

## Running the notebooks

### JupyterLab in Docker (recommended)

Everything runs on the stock `quay.io/jupyter/datascience-notebook:latest` image (Python 3, numpy, scipy, pandas,
matplotlib, seaborn, PyYAML, R):

```bash
cd docker
NOTEBOOK_TOKEN=choose-a-token docker compose up -d
```

Then open <http://localhost:8899> and browse to `work/`. The compose file mounts the repo at `/home/jovyan/work`.

- Notebooks that use a project package (`checkout_queues`, `time_to_traffic_sim`, `chip-bets`' modules) add their
  project folder to `sys.path` themselves, so there is nothing to install. Jupyter starts each kernel in the
  notebook's own folder, which is all they rely on.
- The R notebooks install the few course packages they need from CRAN on first use in a fresh container.
- `docker/Dockerfile` builds an optional image with those R packages and CuPy preinstalled; switch to it with
  `build: .` in `compose.yaml`.

### Locally

```bash
pip install jupyterlab numpy scipy pandas matplotlib seaborn pyyaml
jupyter lab
```

The R notebooks additionally need R with the IRkernel.

### Tests

The two Python packages have pytest suites. pytest is not in the Jupyter image, so `pip install pytest` first, then
run `python -m pytest` in `checkout-queues/` or `time-to-traffic-metric/`.

## Conventions

The notebooks use the same small set of libraries in the same way:

- `numpy` for arrays and simulation, with a seeded generator, `rng = np.random.default_rng(seed)`, so reruns
  reproduce the saved outputs. Simulations draw all trials as arrays rather than looping in Python where they can.
- `pandas` for tables, `scipy.stats` for distributions and tests.
- `matplotlib.pyplot as plt` for plots; `seaborn` only where it adds something (faceted or violin/KDE plots, and
  its example datasets).
- Code shared between notebooks, or long enough to obscure a notebook, lives in a module in the project folder,
  with tests.

## Performance and the GPU

Measured on a 4-CPU machine, executing every Python notebook from scratch took about 157 s before this cleanup and
about 63 s after (roughly 1 s of each is kernel start-up). The gains came from replacing Python loops with numpy
array operations (birthday paradox 20.6 s → 2.3 s, puzzle pieces 24 s → 1.5 s with 7× more trials, hiring problem
13.7 s → 1.8 s) and from running independent simulations in parallel processes (chip-bets 48.7 s → 13 s, which
scales further with more cores).

The RTX 2080 Ti does not speed up any notebook at its current size: every simulation now finishes in well under a
second of array work, and moving data to the GPU and initializing CUDA would cost more than it saves. The two
CPU-bound notebooks that remain (`chip-bets`, the power-of-d object model) are sequential Python logic, which a
GPU cannot run. The one place where a GPU makes sense is a much larger Monte Carlo run in `time-to-traffic-metric`,
which has an optional CuPy backend (`simulation.backend: auto` in its `config.yaml`). To use it:

1. install the NVIDIA Container Toolkit on the host,
2. uncomment the `deploy:` GPU block in `docker/compose.yaml`,
3. install CuPy in the container (`pip install "cupy-cuda12x[ctk]"`, or build `docker/Dockerfile`).

## License

MIT — see [LICENSE](LICENSE).
