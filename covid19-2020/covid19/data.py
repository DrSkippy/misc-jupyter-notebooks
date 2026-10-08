"""The 2020-21 COVID-19 case data used by the notebooks.

Both original sources have stopped: the COVID Tracking Project ended on 2021-03-07, and Our World in Data now
publishes WHO's weekly totals instead of the daily series the notebooks were written against. The data is kept
here as compressed snapshots, built by `refresh_snapshots()` from the projects' archives on GitHub:

- US states, daily: COVID Tracking Project API archive (COVID19Tracking/covid-public-api)
- countries, daily: Johns Hopkins CSSE time series (CSSEGISandData/COVID-19), summed over provinces

Both snapshots end on 2021-03-07. Every loader takes `as_of` to cut the data at a date, so a notebook can show the
picture as it looked at the time.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
US_SNAPSHOT = DATA_DIR / "us_states_daily.csv.gz"
WORLD_SNAPSHOT = DATA_DIR / "world_daily.csv.gz"
POPULATION_FILE = DATA_DIR / "nst-est2019-01.csv"  # Census 2019 state population estimates

CTP_URL = "https://raw.githubusercontent.com/COVID19Tracking/covid-public-api/master/v1/states/daily.csv"
JHU_URL = ("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/"
           "csse_covid_19_time_series/time_series_covid19_{}_global.csv")
SNAPSHOT_END = "2021-03-07"

# Columns every loader returns (US data also has negative, pending, tests and population)
COUNT_COLUMNS = ["positive", "daily_new_positive", "death", "daily_new_death"]


def refresh_snapshots() -> None:
    """Download the archives and rewrite the two snapshot files (about 30 MB of downloads)."""
    us = pd.read_csv(CTP_URL, usecols=["date", "state", "positive", "negative", "pending", "totalTestResults",
                                       "death", "positiveIncrease", "deathIncrease"])
    us["date"] = pd.to_datetime(us["date"].astype(str), format="%Y%m%d")
    us = us.rename(columns={"totalTestResults": "tests", "positiveIncrease": "daily_new_positive",
                            "deathIncrease": "daily_new_death"})
    us.sort_values(["state", "date"]).to_csv(US_SNAPSHOT, index=False, date_format="%Y-%m-%d")

    frames = []
    for kind, column in (("confirmed", "positive"), ("deaths", "death")):
        wide = pd.read_csv(JHU_URL.format(kind)).drop(columns=["Province/State", "Lat", "Long"])
        long = (wide.groupby("Country/Region").sum().stack().rename(column).rename_axis(["country", "date"]))
        frames.append(long)
    world = pd.concat(frames, axis=1).reset_index()
    world["date"] = pd.to_datetime(world["date"], format="%m/%d/%y")
    world = world[world["date"] <= SNAPSHOT_END].sort_values(["country", "date"])
    world.to_csv(WORLD_SNAPSHOT, index=False, date_format="%Y-%m-%d")


def state_populations() -> pd.DataFrame:
    """Columns state (two-letter code) and population."""
    return pd.read_csv(POPULATION_FILE, thousands=",")[["state", "population"]]


def _rank(df: pd.DataFrame) -> list[str]:
    """Places ordered by total positive cases on the last date, largest first."""
    last = df.sort_values("date").groupby("state").tail(1)
    return last.sort_values("positive", ascending=False)["state"].tolist()


def us_states_daily(as_of: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    """US states and territories, one row per (state, date), and the states ranked by positive cases.

    Columns: date, state, positive, negative, pending, tests, death, daily_new_positive, daily_new_death,
    population. Missing case and death counts are 0; missing test counts stay NaN.
    """
    df = pd.read_csv(US_SNAPSHOT, parse_dates=["date"])
    if as_of is not None:
        df = df[df["date"] <= as_of]
    df[COUNT_COLUMNS + ["negative", "pending"]] = df[COUNT_COLUMNS + ["negative", "pending"]].fillna(0)
    df = df.merge(state_populations(), on="state", how="left")
    return df.reset_index(drop=True), _rank(df)


def world_daily(as_of: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Countries (JHU names, e.g. "Italy", "Korea, South"), one row per (country, date), and the countries ranked
    by positive cases.

    Columns: date, state (the country), positive, death, daily_new_positive, daily_new_death.
    """
    df = pd.read_csv(WORLD_SNAPSHOT, parse_dates=["date"]).rename(columns={"country": "state"})
    if as_of is not None:
        df = df[df["date"] <= as_of]
    df = df.sort_values(["state", "date"]).reset_index(drop=True)
    by_country = df.groupby("state")
    df["daily_new_positive"] = by_country["positive"].diff().fillna(0)
    df["daily_new_death"] = by_country["death"].diff().fillna(0)
    return df, _rank(df)


def get_state_df(df: pd.DataFrame, state: str, additional_aggregation_keys: list[str] | None = None) -> pd.DataFrame:
    """Rows for one state or country; "*" sums every place by date (the US total for US data).

    For "*", the count columns plus any `additional_aggregation_keys` (and `tests`, if present) are summed.
    """
    if state != "*":
        return df.loc[df["state"] == state].reset_index(drop=True)
    keys = COUNT_COLUMNS + (["tests"] if "tests" in df else []) + list(additional_aggregation_keys or [])
    return df.groupby("date", as_index=False)[keys].sum(min_count=1)
