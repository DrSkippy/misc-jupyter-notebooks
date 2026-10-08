# COVID-19 in 2020: doubling times and SIR fits

Naive, short-term views of COVID-19 case growth from spring to fall 2020: exponential fits and doubling times for
US states and selected countries, how those doubling times changed, testing rates, and SIR model fits to the
first wave. Moved here from the `DrSkippy/2020covid19` repository.

## Layout

| Path | Contents |
|---|---|
| `covid19/data.py` | Loads the case data (US states and countries), with `as_of` to cut it at a date; aggregation helper |
| `covid19/models.py` | Doubling-time fits, naive case and hospital-load estimates, exponential projection, SIR and SIR4 models |
| `notebooks/comparisons.ipynb` | US states and countries as of 2020-09-23: growth curves, doubling times, rolling doubling periods, testing |
| `notebooks/sir-fits.ipynb` | SIR and SIR4 fits to the first wave (to 2020-05-01) in the US, Italy and Spain |
| `notebooks/presentation-2020-04-01.ipynb` | A brown-bag talk given on 2020-04-01, run on the data available that morning |
| `data/` | Data snapshots (below) and Census 2019 state population estimates |
| `tests/` | pytest suite |

The notebooks add the project folder to `sys.path` themselves, so they run on the stock Jupyter image without
installing anything. Tests: `pip install pytest`, then `python -m pytest` in this folder.

## Data

The notebooks originally pulled live data from covidtracking.com and Our World in Data. Neither works any more:
the COVID Tracking Project stopped on 2021-03-07, and OWID now publishes WHO's weekly totals, which are flat for a
week and then jump, so daily doubling times can't be computed from them. The data is kept here instead, as two
compressed snapshots (about 0.8 MB) built from archives on GitHub:

| Snapshot | Source | Coverage |
|---|---|---|
| `data/us_states_daily.csv.gz` | COVID Tracking Project API archive (`COVID19Tracking/covid-public-api`) | 56 states and territories, 2020-01-13 to 2021-03-07 |
| `data/world_daily.csv.gz` | Johns Hopkins CSSE time series (`CSSEGISandData/COVID-19`), summed over provinces | countries, 2020-01-22 to 2021-03-07 |

`covid19.data.refresh_snapshots()` rebuilds both from those archives. Countries are named as JHU names them
(`"Italy"`, `"Korea, South"`, `"United Kingdom"`); the original notebooks used ISO codes from OWID.

Each notebook sets `AS_OF`, so it shows the picture as it looked when it was written; move it to see another date.

## Models

- **Doubling time**: a straight-line fit to log(total cases) over the last *n* days; ln 2 / slope.
- **Rolling doubling period / resolution time**: the doubling time over each 10-day window, divided by a ~10-day
  time for a case to resolve. Below 1, active cases grow.
- **SIR**: dS/dt = −βSI/N, dI/dt = βSI/N − γI, dR/dt = γI on a daily grid. While almost everyone is
  susceptible, cumulative cases only determine β − γ, so fits fix γ = 0.1 per day (a 10-day infectious period) and
  fit β and the initial infections, by least squares on log cumulative cases. Early on, cases double every
  ln 2 / (β − γ) days, and R₀ = β/γ.
- **SIR4**: adds dβ/dt = −αβI, so the contact rate falls as infections rise.
- **Cases from deaths**: early testing missed most cases, so the SIR fits infer cases from deaths (each death
  stands for 200-250 infections that started 10-14 days earlier).

## Changes from the 2020 repository

Fixes:

- Every module configured logging to a hard-coded file (`/home/scott/log/covid.log`, `/Users/drskippy/logs/...`),
  so importing failed on any other machine. Paths to data files were also hard-coded or relative to the working
  directory.
- The SIR model integrated on `linspace(0, periods, periods)`, so its "days" were slightly longer than a day, and
  `SIR.project` ignored its `periods` argument (any value but 160 failed).
- The exponential projection evaluated the fit one day ahead of the fit's own time axis.
- The world data's daily new cases were differenced across country boundaries (each country's first day got the
  previous country's last total subtracted).
- Aggregating world data (`"*"`) failed on the missing `tests` column; `int()` of a NaN estimate crashed the
  hospital and ICU estimates; the doubling-time fit modified the caller's data in place and trimmed trailing as
  well as leading zeros, which misaligned the fit.
- SIR fits drifted to degenerate solutions (1/γ of minutes, R₀ in the millions). They now fix γ and fit on a log
  scale, as described above. In `SIR Fits.ipynb`, the Spain fit reused Italy's data, the SIR4 summaries printed the
  previous model's β, and "doubling time ≈ ln 2/β, recovery time ≈ ln 2/γ" are now ln 2/(β − γ) and 1/γ.
- SIR4's β could go negative (dβ/dt = −αI); it is now dβ/dt = −αβI.
- The presentation notebook was written against an earlier version of the code (functions returning tuples,
  `SIRFitter` as a function, `estimate_current_cases`) and could not run; it is ported to the current code.

Cleanup: the three packages (`covid_tracking_data`, `ourworldindata_org`, `covid_analysis`) are now one,
`covid19`; Poetry and `.envrc` are replaced by a plain `pyproject.toml`; `comparisons.ipynb` drew about 450
separate figures (11 MB), mostly one set per state, and now uses a doubling-time table and small-multiple grids.
