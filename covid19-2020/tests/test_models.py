import numpy as np
import pandas as pd
import pytest

from covid19.data import get_state_df, us_states_daily, world_daily
from covid19.models import (
    SIR,
    SIR4,
    CurrentCasesFromDeaths,
    ModelProjectionExponential,
    doubling_time_in_days,
    get_zero_aligned_log_positive,
    rolling_doubling_period,
)


def exponential_df(days=30, doubling=4.0, start=10.0, lead_zeros=0):
    t = np.arange(days)
    positive = np.concatenate([np.zeros(lead_zeros), start * 2 ** (t / doubling)])
    return pd.DataFrame({"date": pd.date_range("2020-03-01", periods=len(positive)), "positive": positive})


def test_doubling_time_recovers_exponential():
    df = exponential_df(doubling=4.0, lead_zeros=5)
    dt, line, (m, b) = doubling_time_in_days(df["positive"].values)
    assert dt == pytest.approx(4.0)
    np.testing.assert_allclose(np.exp(line[5:]), df["positive"].values[5:], rtol=1e-9)


def test_doubling_time_uses_last_n_days_and_does_not_modify_input():
    y = np.concatenate([10 * 2 ** (np.arange(10) / 2.0), 10 * 2 ** (4.5 + np.arange(10) / 8.0)])
    y_with_glitch = y.copy()
    y_with_glitch[3] = 0  # spurious zero
    before = y_with_glitch.copy()
    assert doubling_time_in_days(y_with_glitch, use_last_n_days=10)[0] == pytest.approx(8.0)
    np.testing.assert_array_equal(y_with_glitch, before)


def test_doubling_time_too_few_points_is_nan():
    assert np.isnan(doubling_time_in_days([0, 0, 5])[0])


def test_exponential_projection_continues_the_fit_without_offset():
    df = exponential_df(days=20, doubling=5.0, start=1000.0)
    proj = ModelProjectionExponential().project(df, days=5)
    assert len(proj) == 25
    expected = (1000.0 * 2 ** (np.arange(25) / 5.0)).astype(int)
    np.testing.assert_allclose(proj["positive_predicted"], expected, rtol=1e-6, atol=1)


def test_sir_conserves_population_on_a_daily_grid():
    df = SIR().SIRModel(N=1000, I0=1, beta=0.4, gamma=0.1, periods=50)
    assert len(df) == 50
    np.testing.assert_allclose(df.sum(axis=1), 1000)


def test_sir_fit_recovers_parameters():
    truth = SIR().SIRModel(N=1000, I0=0.5, beta=0.35, gamma=0.1, periods=60)
    c = (truth.infected + truth.removed).values
    N, I0, R0, beta, gamma = SIR().SIRFitter(c, N=1000, gamma=0.1)
    assert beta == pytest.approx(0.35, rel=1e-3)
    assert I0 == pytest.approx(0.5, rel=1e-2)


def test_sir4_fit_recovers_parameters_and_beta_stays_positive():
    truth = SIR4().SIRModel(N=1000, I0=0.5, beta0=0.4, alpha=0.02, gamma=0.1, periods=60)
    assert (truth.beta > 0).all() and truth.beta.iloc[-1] < 0.4
    c = (truth.infected + truth.removed).values
    N, I0, R0, beta0, alpha, gamma = SIR4().SIRFitter(c, N=1000, gamma=0.1)
    assert beta0 == pytest.approx(0.4, rel=1e-2)
    assert alpha == pytest.approx(0.02, rel=5e-2)


def test_sir_project_respects_periods():
    proj = SIR().project(np.arange(30.0), periods=90,
                         params={"SIR": (1000, 1, 0, 0.3, 0.1), "start_date": "2020-03-01"})
    assert len(proj) == 90 and proj["date"].iloc[-1] == pd.Timestamp("2020-05-29")
    proj4 = SIR4().project(np.arange(30.0), periods=90,
                           params={"SIR": (1000, 1, 0, 0.3, 0.001, 0.1), "start_date": "2020-03-01"})
    assert len(proj4) == 90


def test_cases_from_deaths_shift():
    df = pd.DataFrame({"daily_new_death": [0, 1, 2, 3, 4]})
    out = CurrentCasesFromDeaths().add_positive_estimate(df, params={"a": 10, "shift": 2})
    assert out["positive_fromdeath"].tolist() == [30, 60, 100, 0, 0]


def test_zero_aligned_starts_at_first_change():
    df = pd.DataFrame({"positive": [5, 20, 20, 20, 40, 80]})
    out = get_zero_aligned_log_positive(df, min_pos=10)
    assert out["positive"].tolist() == [20, 40, 80]
    np.testing.assert_allclose(out["log_positive"], np.log([1, 2, 4]))
    assert get_zero_aligned_log_positive(pd.DataFrame({"positive": [20, 20]}), min_pos=10).empty


def test_rolling_doubling_period():
    dates, periods = rolling_doubling_period(exponential_df(days=20, doubling=3.0), window_size=10)
    assert len(periods) == 11
    np.testing.assert_allclose(periods, 3.0)


def test_snapshots_load_and_aggregate():
    us, states = us_states_daily(as_of="2020-04-01")
    assert us["date"].max() == pd.Timestamp("2020-04-01")
    assert states[0] == "NY"
    total = get_state_df(us, "*")
    assert total["positive"].iloc[-1] == us[us["date"] == "2020-04-01"]["positive"].sum()

    world, countries = world_daily(as_of="2020-04-01")
    italy = get_state_df(world, "Italy")
    assert (italy["daily_new_positive"].iloc[1:].values == np.diff(italy["positive"].values)).all()
    # daily differences restart for each country instead of running across the boundary
    assert (world.groupby("state")["daily_new_positive"].first() == 0).all()
    assert "tests" not in get_state_df(world, "*")
