"""How long until a coin-toss pattern first appears?

Rewrite of 2008 Python 2 scripts. Tosses are strings of 'H' and 'T'.

- ``simulate_waiting_times``: Monte Carlo number of tosses until a pattern first appears
- ``count_without_pattern``: exact number of length-n sequences that never contain a pattern
- ``survival_table`` / ``waiting_time_pmf``: the same counts as probabilities
- ``expected_tosses``: exact mean waiting time (Conway's overlap formula)
"""
from itertools import product

import numpy as np
import pandas as pd


def all_sequences(n):
    """All 2**n toss sequences of length n, as strings."""
    return ["".join(s) for s in product("HT", repeat=n)]


def _check(pattern):
    if not pattern or set(pattern) - {"H", "T"}:
        raise ValueError(f"pattern must be a non-empty string of 'H' and 'T', got {pattern!r}")


def simulate_waiting_times(pattern, n_trials=20_000, rng=None, block=64):
    """Number of tosses until ``pattern`` first appears, for each of ``n_trials`` independent runs.

    Tosses are bits (H=1, T=0); the last len(pattern) tosses of every run are kept as an integer and compared
    to the pattern's code. All runs advance together, ``block`` tosses at a time, until every run has matched.
    """
    _check(pattern)
    rng = np.random.default_rng(rng)
    k = len(pattern)
    target = int(pattern.replace("H", "1").replace("T", "0"), 2)
    mask = (1 << k) - 1

    waits = np.zeros(n_trials, dtype=np.int64)
    active = np.arange(n_trials)
    state = np.zeros(n_trials, dtype=np.int64)
    tossed = 0
    while active.size:
        flips = rng.integers(0, 2, size=(active.size, block))
        codes = np.empty_like(flips)
        for j in range(block):
            state = ((state << 1) | flips[:, j]) & mask
            codes[:, j] = state
        hits = codes == target
        hits[:, : max(0, k - 1 - tossed)] = False  # fewer than k tosses so far: leading zeros are not tails
        found = hits.any(axis=1)
        waits[active[found]] = tossed + hits[found].argmax(axis=1) + 1
        active, state = active[~found], state[~found]
        tossed += block
    return waits


def waiting_time_counts(waits):
    """Number of runs that ended on each toss count, with zeros filled in between min and max."""
    counts = pd.Series(waits).value_counts().sort_index()
    counts = counts.reindex(range(counts.index.min(), counts.index.max() + 1), fill_value=0)
    counts.index.name = "tosses"
    return counts.rename("runs")


def _transitions(pattern):
    """KMP automaton: state j = length of the longest pattern prefix ending the sequence so far."""
    k = len(pattern)
    table = []
    for j in range(k):
        row = {}
        for c in "HT":
            s = pattern[:j] + c
            row[c] = next(m for m in range(len(s), -1, -1) if pattern[:m] == s[len(s) - m:])
        table.append(row)
    return table


def count_without_pattern(n, pattern):
    """Exact number of length-n toss sequences that do not contain ``pattern``."""
    _check(pattern)
    k = len(pattern)
    table = _transitions(pattern)
    counts = [1] + [0] * (k - 1)  # python ints, so large n stays exact
    for _ in range(n):
        nxt = [0] * k
        for j, c in enumerate(counts):
            for m in table[j].values():
                if m < k:
                    nxt[m] += c
        counts = nxt
    return sum(counts)


def sequences_without_pattern(n, pattern):
    """Brute force: every length-n sequence that does not contain ``pattern`` (keep n small)."""
    _check(pattern)
    return [s for s in all_sequences(n) if pattern not in s]


def survival_table(patterns, n_max):
    """Fraction of length-n sequences without each pattern, n = 1..n_max (one column per pattern).

    This is also P(waiting time > n).
    """
    n = range(1, n_max + 1)  # python ints: 2**n overflows int64 past n = 62
    table = pd.DataFrame({p: [count_without_pattern(i, p) / 2**i for i in n] for p in patterns}, index=n)
    table.index.name = "tosses"
    return table


def waiting_time_pmf(pattern, n_max):
    """Exact P(pattern first appears on toss n), n = 1..n_max: P(N > n-1) - P(N > n)."""
    survival = pd.concat([pd.Series([1.0], index=[0]), survival_table([pattern], n_max)[pattern]])
    pmf = (survival.shift(1) - survival).iloc[1:]
    pmf.index.name = "tosses"
    return pmf.rename(pattern)


def expected_tosses(pattern):
    """Exact mean number of tosses to see ``pattern``: sum of 2**j over every j where the length-j prefix
    equals the length-j suffix (Conway). E.g. HH -> 2 + 4 = 6, HT -> 4."""
    _check(pattern)
    return sum(2**j for j in range(1, len(pattern) + 1) if pattern[:j] == pattern[-j:])
