# Shared setup for the course notebooks; each notebook starts with source("setup.R").

# Load packages, installing any that are missing from CRAN first. The stock Jupyter image has R but
# not these course packages, so the first notebook run in a fresh container installs them.
use_packages <- function(...) {
  pkgs <- c(...)
  missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing)) install.packages(missing, repos = "https://cloud.r-project.org")
  for (p in pkgs) suppressPackageStartupMessages(library(p, character.only = TRUE))
}

# Right heart catheterization (RHC) data, reduced to the variables used in the lectures.
# Downloaded once to data/rhc.sav.
#   treatment: 1 = RHC (swang1), died: 1 = death, plus disease-category dummies, age, female, meanbp1
load_rhc <- function(path = "data/rhc.sav") {
  if (!file.exists(path)) download.file("https://hbiostat.org/data/repo/rhc.sav", path, mode = "wb")
  env <- new.env()
  load(path, envir = env)
  rhc <- env$rhc
  data.frame(
    ARF       = as.numeric(rhc$cat1 == "ARF"),
    CHF       = as.numeric(rhc$cat1 == "CHF"),
    Cirr      = as.numeric(rhc$cat1 == "Cirrhosis"),
    colcan    = as.numeric(rhc$cat1 == "Colon Cancer"),
    Coma      = as.numeric(rhc$cat1 == "Coma"),
    lungcan   = as.numeric(rhc$cat1 == "Lung Cancer"),
    MOSF      = as.numeric(rhc$cat1 == "MOSF w/Malignancy"),
    sepsis    = as.numeric(rhc$cat1 == "MOSF w/Sepsis"),
    age       = rhc$age,
    female    = as.numeric(rhc$sex == "Female"),
    meanbp1   = rhc$meanbp1,
    treatment = as.numeric(rhc$swang1 == "RHC"),
    died      = as.numeric(rhc$death == "Yes")
  )
}

# Covariates used in the RHC examples (a shorter list than you would use in practice)
rhc_xvars <- c("ARF", "CHF", "Cirr", "colcan", "Coma", "lungcan", "MOSF", "sepsis", "age", "female", "meanbp1")

logit <- function(p) log(p) - log(1 - p)
expit <- function(x) 1 / (1 + exp(-x))

# Rows of `data` kept by a Matching::Match result: treated rows, then their matched controls
matched_rows <- function(match, data) data[unlist(match[c("index.treated", "index.control")]), ]
