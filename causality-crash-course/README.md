# A Crash Course in Causality

R notebooks working through the University of Pennsylvania course *A Crash Course in Causality: Inferring Causal
Effects from Observational Data* (Coursera): lecture examples and the data projects.

| Notebook | Method | Data |
|---|---|---|
| `1-matching-rhc.ipynb` | Mahalanobis and propensity score matching, McNemar test | Right heart catheterization (RHC) |
| `2-ipw-rhc.ipynb` | Inverse probability weighting, marginal structural models, weight truncation | RHC |
| `3-instrumental-variables-card.ipynb` | Instrumental variables, complier average causal effect, 2SLS | Card's college proximity data (`data/card.data.rda`) |
| `p1-lalonde-matchit.ipynb` | Propensity score matching, with and without a caliper | Lalonde (`MatchIt` version) |
| `p2-lalonde-matching.ipynb` | The same analysis on the other Lalonde sample | Lalonde (`Matching` version) |
| `p3-lalonde-ipw.ipynb` | Inverse probability weighting and an MSM | Lalonde (`MatchIt` version) |

Every notebook starts with `source("setup.R")`, which provides

- `use_packages(...)`: loads packages, installing any that are missing from CRAN first. The stock Jupyter image
  has R but not these packages, so the first run in a fresh container takes a minute longer. To bake them in,
  use `docker/Dockerfile`.
- `load_rhc()`: downloads the RHC data once to `data/rhc.sav` (not committed) and builds the analysis data frame.
- `matched_rows()`, `logit()`, `expit()`: small helpers used across notebooks.

Notes

- Two packages ship different datasets named `lalonde`; each notebook loads one explicitly with
  `data("lalonde", package = ...)`.
- `ivpack`, used in the course for 2SLS, has been removed from CRAN; notebook 3 uses `fixest` instead.
- The RHC notebooks (1 and 2) are committed without outputs; run them to download the data and fill them in.
