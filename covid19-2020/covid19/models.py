"""Growth-rate fits, naive case estimates, exponential projections and SIR models."""

from __future__ import annotations

import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import ODEintWarning, odeint
from scipy.optimize import minimize

from covid19.data import get_state_df

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------------------------------------------
# Doubling time


def _fix_spurious_zeros(x: np.ndarray) -> np.ndarray:
    """Replace a zero after a non-zero total (a reporting glitch, e.g. NV) with the previous value.

    Assumes leading zeros were already trimmed. Returns a copy.
    """
    x = np.array(x, dtype=float)
    for i in range(1, len(x)):
        if x[i] == 0:
            log.info("fixing spurious zero at index=%d", i)
            x[i] = x[i - 1]
    return x


def doubling_time_in_days(total_by_day, use_last_n_days=None):
    """Fit log(total) = m t + b to a daily cumulative series (leading zeros dropped).

    Returns (doubling time ln 2 / m in days, fitted curve log values for every day, (m, b)). A negative doubling
    time means the total is shrinking; with fewer than two usable points the result is NaN.
    """
    total_by_day = np.asarray(total_by_day, dtype=float)
    y = _fix_spurious_zeros(np.trim_zeros(total_by_day, "f"))
    if use_last_n_days is not None:
        y = y[-int(use_last_n_days):]
    t_all = np.arange(len(total_by_day))
    if len(y) < 2:
        return np.nan, np.full(len(total_by_day), np.nan), (np.nan, np.nan)
    m, b = np.polyfit(t_all[-len(y):], np.log(y), 1)
    doubling = np.log(2) / m if m != 0 else np.inf
    return doubling, m * t_all + b, (m, b)


def get_doubling_df(df_res, key="positive", use_last_n_days=None):
    """Doubling-time fit of `key`; returns (copy of df_res with an exp_fit_line column, doubling time, (m, b))."""
    df_res = df_res.copy()
    dt, line, params = doubling_time_in_days(df_res[key].values, use_last_n_days)
    df_res["exp_fit_line"] = np.exp(line)
    return df_res, dt, params


def get_state_doubling_df(df, state, key="positive", use_last_n_days=None):
    return get_doubling_df(get_state_df(df, state), key, use_last_n_days)


def get_zero_aligned_log_positive(df, min_pos=10, key="positive", use_last_n_days=None):
    """Rows from the first day with more than `min_pos` cases (and a change from the day before), with
    log_<key> relative to that day and a days_since_<min_pos> counter. Empty if there are not two such days."""
    df_res = df.loc[df[key] > min_pos].copy()
    if use_last_n_days is not None:
        df_res = df_res.iloc[-use_last_n_days:]
    values = df_res[key].values
    first_change = np.flatnonzero(np.diff(values) != 0)
    df_res = df_res.iloc[first_change[0]:] if len(first_change) else df_res.iloc[:0]
    log_key = f"log_{key}"
    df_res[log_key] = np.log(df_res[key]) - (np.log(df_res[key].iloc[0]) if len(df_res) else 0)
    df_res[f"days_since_{min_pos}"] = np.arange(len(df_res))
    return df_res


# ---------------------------------------------------------------------------------------------------------------
# Naive current-case estimates


def _as_int(series: pd.Series) -> pd.Series:
    return series.fillna(0).astype(int)


class CurrentCases:
    """Naive estimates layered on a daily data frame. Each add_* method returns the frame with a new column."""

    def add_positive_estimate(self, df, params=None, suffix="ident", base="positive"):
        params = params or {"a": 1}
        df[f"positive_{suffix}"] = df[base] * params["a"]
        return df

    def add_death_estimate(self, df, params=None, suffix="ident", base="death"):
        params = params or {"a": 1}
        df[f"death_{suffix}"] = _as_int(df[base] * params["a"])
        return df

    def add_hospitalized_estimate(self, df, params=None, suffix="ident", base="new_daily_positive"):
        """Fraction `a` of the cases from the last `rt` days."""
        params = params or {"a": .15, "rt": 10}
        df[f"hospitalized_{suffix}"] = _as_int(df[base].rolling(params["rt"]).sum() * params["a"])
        return df

    def add_icu_estimate(self, df, params=None, suffix="ident", base="new_daily_positive"):
        """Fraction `a` of the cases from the last `rt` days."""
        params = params or {"a": .04, "rt": 10}
        df[f"icu_{suffix}"] = _as_int(df[base].rolling(params["rt"]).sum() * params["a"])
        return df


class CurrentCasesFromDeaths(CurrentCases):
    """Infer cases from deaths: `a` cases per death, infected `shift` days before dying."""

    def add_positive_estimate(self, df, params=None, suffix="fromdeath", base="daily_new_death"):
        params = params or {"a": 200, "shift": 10}
        daily = df[base] * params["a"]
        df[f"daily_new_positive_{suffix}"] = daily
        total = _as_int(daily.cumsum())
        self.fit_series = total.values.copy()
        self.shift = params["shift"]
        df[f"positive_{suffix}"] = total.shift(-params["shift"], fill_value=0)
        return df


class CurrentCasesUndercount(CurrentCases):
    """Scale reported cases up by (1 + a) for the cases testing misses."""

    def add_positive_estimate(self, df, params=None, suffix="undercount", base="daily_new_positive"):
        params = params or {"a": 0.8}
        daily = df[base] * (1 + params["a"])
        df[f"daily_new_positive_{suffix}"] = daily
        df[f"positive_{suffix}"] = _as_int(daily.cumsum())
        self.fit_series = df[f"positive_{suffix}"].values
        self.shift = 0
        return df


def estimate_current_cases(daily_new_positive, resolution_days=10, hospitalized=0.15, icu=0.04):
    """(unresolved cases, hospitalized, ICU): cases from the last `resolution_days` days and fixed fractions."""
    current = float(np.sum(np.asarray(daily_new_positive)[-resolution_days:]))
    return int(current), int(current * hospitalized), int(current * icu)


def _add_care_estimates(df):
    df["new_daily_positive"] = df["positive_predicted"].diff().fillna(0)
    cc = CurrentCases()
    return cc.add_icu_estimate(cc.add_hospitalized_estimate(df))


# ---------------------------------------------------------------------------------------------------------------
# Projections


class ModelProjectionExponential:
    """Extend the exponential fit of the last `last_n_days` days `days` into the future."""

    def project(self, df, days, params=None):
        params = params or {"last_n_days": 10}
        _, _, (m, b) = get_doubling_df(df, key="positive", use_last_n_days=params["last_n_days"])
        periods = len(df) + days
        t = np.arange(periods)  # same time base as the fit: day 0 is the first row of df
        positive = np.zeros(periods)
        positive[:len(df)] = df["positive"].values
        res = pd.DataFrame({
            "day_number": t,
            "date": pd.date_range(start=df["date"].values[0], periods=periods),
            "positive_predicted": np.exp(b + m * t).astype(int),
            "positive": positive,
        })
        return _add_care_estimates(res)


class SIR:
    """Susceptible-infected-removed model on a daily grid, and fits of it to cumulative cases.

    Populations are in whatever unit the caller uses (the notebooks use thousands of people).

    Fitting: while S ≈ N, cumulative cases only pin down the growth rate β - γ, not β and γ separately; a free
    fit drifts to β ≈ γ → ∞. So γ (1 / mean infectious period) is fixed, by default to 0.1 per day, and the fit
    finds the rest by minimizing squared differences of log cumulative cases (each day counts by its relative
    error, not its size). Parameters are fit on a log scale so they stay positive.
    """

    def SIRModel(self, N=100, I0=1, R0=0, beta=0.3, gamma=0.1, periods=160):
        """Integrate dS/dt = -βSI/N, dI/dt = βSI/N - γI, dR/dt = γI for `periods` days (one row per day)."""
        def deriv(y, t, N, beta, gamma):
            S, I, R = y
            return -beta * S * I / N, beta * S * I / N - gamma * I, gamma * I

        S, I, R = odeint(deriv, (N - I0 - R0, I0, R0), np.arange(periods), args=(N, beta, gamma)).T
        return pd.DataFrame({"susceptible": S, "infected": I, "removed": R})

    def _model_kwargs(self, x):
        """Fit vector -> SIRModel keyword arguments (besides N, R0, gamma, periods)."""
        log_beta, log_I0 = x
        return dict(beta=np.exp(log_beta), I0=np.exp(log_I0))

    def _x0(self, c, gamma):
        """Starting point: growth rate r of the first 10 days, so β = r + γ, and I0 = the first value."""
        n = min(10, len(c))
        r = np.polyfit(np.arange(n), np.log(c[:n]), 1)[0]
        return [np.log(max(r + gamma, 1e-3)), np.log(c[0])]

    def SIRSSE(self, x, N, R0, c, gamma):
        """Squared error between log cumulative cases c and log of the model's infected + removed."""
        dfp = self.SIRModel(N, R0=R0, gamma=gamma, periods=len(c), **self._model_kwargs(x))
        predicted = dfp.infected.values + dfp.removed.values
        if not np.all(np.isfinite(predicted)):
            return np.inf
        return np.sum((np.log(c) - np.log(np.maximum(predicted, 1e-12))) ** 2)

    def SIRFitter(self, c, N=350000, gamma=0.1, R0=0):
        """Fit to cumulative cases c (all > 0); returns the arguments for SIRModel/project, in their order:
        (N, I0, R0, beta, gamma)."""
        c = np.asarray(c, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ODEintWarning)
            self.fit_result = minimize(self.SIRSSE, self._x0(c, gamma), args=(N, R0, c, gamma),
                                       method="Nelder-Mead", options={"maxiter": 4000})
        log.info("%s fit: %s", type(self).__name__, self.fit_result.message)
        kw = self._model_kwargs(self.fit_result.x)
        return self._fitted(N, R0, gamma, kw)

    def _fitted(self, N, R0, gamma, kw):
        return (N, kw["I0"], R0, kw["beta"], gamma)

    def project(self, c, periods, params):
        """Model run of `periods` days from params["SIR"] (a fitted tuple) starting at params["start_date"],
        with the observed cumulative cases c, care estimates, and dates."""
        dfm = self.SIRModel(*params["SIR"], periods=periods)
        dfm["date"] = pd.date_range(start=params["start_date"], periods=periods)
        observed = np.zeros(periods)
        observed[:len(c)] = c[:periods]
        dfm["positive"] = observed
        dfm["positive_predicted"] = dfm.infected + dfm.removed
        return _add_care_estimates(dfm)


class SIR4(SIR):
    """SIR with a fourth equation for a contact rate that changes with the number infected: dβ/dt = -αβI.

    With α > 0, β decays as infections rise (distancing, lockdowns) and never goes negative. (The 2020 version
    used dβ/dt = -αI, which fits drive below zero.)
    """

    def SIRModel(self, N=100, I0=1, R0=0, beta0=0.3, alpha=0.0, gamma=0.1, periods=160):
        def deriv(y, t, N, alpha, gamma):
            S, I, R, beta = y
            return -beta * S * I / N, beta * S * I / N - gamma * I, gamma * I, -alpha * beta * I

        S, I, R, beta = odeint(deriv, (N - I0 - R0, I0, R0, beta0), np.arange(periods), args=(N, alpha, gamma)).T
        return pd.DataFrame({"susceptible": S, "infected": I, "removed": R, "beta": beta})

    def _model_kwargs(self, x):
        alpha, log_beta0, log_I0 = x
        return dict(alpha=alpha, beta0=np.exp(log_beta0), I0=np.exp(log_I0))

    def _x0(self, c, gamma):
        return [0.0] + super()._x0(c, gamma)

    def _fitted(self, N, R0, gamma, kw):
        """(N, I0, R0, beta0, alpha, gamma)"""
        return (N, kw["I0"], R0, kw["beta0"], kw["alpha"], gamma)


# ---------------------------------------------------------------------------------------------------------------
# Plots


def rolling_doubling_period(dfq, window_size=10):
    """Doubling period of `positive` fitted over each `window_size`-day window; returns (window end dates, periods)."""
    dates, periods = [], []
    for end in range(window_size, len(dfq) + 1):
        window = dfq.iloc[end - window_size:end]
        periods.append(doubling_time_in_days(window["positive"].values, window_size)[0])
        dates.append(window["date"].values[-1])
    return np.array(dates), np.array(periods)


def period_factor_plot(df, code="*", window_size=10, resolution_time=10, ylimit=7, name=None):
    """Rolling doubling period for one place, and its ratio to the resolution (recovery) time.

    A ratio below 1 means cases double faster than they resolve, so active cases grow; the shaded bands mark ratios
    below 3 (red) and 3-5 (yellow). `code` is a state or country, or "*" for the total.
    """
    name = name or ("US" if code == "*" else code)
    dates, periods = rolling_doubling_period(get_state_df(df, code), window_size)
    finite = periods[np.isfinite(periods) & (periods > 0)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[7, 4.5], sharex=True)
    ax1.plot(dates, periods, "*r-")
    ax1.set_ylim(0, min(100, 1.1 * finite.max()) if len(finite) else 100)
    ax1.set_title(f"{name} Doubling Period ({window_size} day moving window)")
    ax1.set_ylabel("Doubling Period")

    ratio = periods / resolution_time
    ax2.plot(dates, ratio, "*b-")
    ax2.axhline(1, color="g")
    ax2.axhspan(0, 3, color="red", alpha=0.1)
    ax2.axhspan(3, 5, color="yellow", alpha=0.1)
    ax2.axhspan(5, ylimit, color="yellow", alpha=0.05)
    ax2.set_ylim(0, ylimit)
    ax2.set_title(f"{name} Ratio ({window_size} day moving window)")
    ax2.set_ylabel("Doubling Period/\nRecovery Time")
    ax2.set_xlabel("Date")
    ax2.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.show()


def describe_sir(beta, gamma, I0, unit=1000):
    """Summary of fitted SIR rates (per day) and initial infections (in `unit` people).

    Early on S ≈ N, so infections grow as exp((β - γ) t): the doubling time is ln 2 / (β - γ).
    """
    growth = beta - gamma
    doubling = f"{np.log(2) / growth:.1f} days" if growth > 0 else "none (not growing)"
    return (f"R0 = beta/gamma = {beta / gamma:.2f}; early doubling time ln2/(beta-gamma) = {doubling}; "
            f"mean infectious period 1/gamma = {1 / gamma:.1f} days; initial infections ~ {unit * I0:,.3g}")
