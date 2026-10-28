"""Population dynamics: strategies grow in proportion to the points they win.

- ``evolve``: the 2008 experiment. A fixed-size population of agents plays a generation of matches, then each
  strategy's share of the next generation is its share of all points scored
- ``apportion``: split a population into whole agents in proportion to weights (largest remainder)
- ``replicator``: the same dynamics for an infinite population, from a mean-score matrix
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .game import PAYOFF, play_totals
from .strategies import get

PAIRINGS = ("round_robin", "random")


@dataclass(frozen=True)
class Evolution:
    """``population``: agents of each strategy per generation (row 0 is the start). ``points``: total points
    each strategy scored in each generation; they decide the next row of ``population``."""
    population: pd.DataFrame
    points: pd.DataFrame

    @property
    def shares(self):
        return self.population.div(self.population.sum(axis=1), axis=0)


def apportion(weights, total):
    """Whole numbers summing to ``total`` in proportion to ``weights`` (Hamilton's largest-remainder method)."""
    weights = np.asarray(weights, dtype=float)
    if total == 0 or weights.sum() == 0:
        return np.zeros(weights.size, dtype=np.int64)
    quotas = weights / weights.sum() * total
    counts = np.floor(quotas).astype(np.int64)
    short = total - counts.sum()
    counts[np.argsort(-(quotas - counts), kind="stable")[:short]] += 1
    return counts


def pair_counts(counts, pairing="round_robin", rng=None):
    """Number of matches between strategies i <= j in one generation, as an upper-triangular matrix.

    ``round_robin``: every agent plays every other agent once. ``random``: agents are shuffled and paired off,
    so each plays one match; with an odd population one agent sits the generation out.
    """
    counts = np.asarray(counts, dtype=np.int64)
    k = counts.size
    if pairing == "round_robin":
        m = np.triu(np.outer(counts, counts), 1)
        m[np.diag_indices(k)] = counts * (counts - 1) // 2
        return m
    if pairing == "random":
        rng = np.random.default_rng(rng)
        agents = rng.permutation(np.repeat(np.arange(k), counts))
        n_pairs = agents.size // 2
        i, j = agents[:n_pairs], agents[n_pairs:2 * n_pairs]
        m = np.zeros((k, k), dtype=np.int64)
        np.add.at(m, (np.minimum(i, j), np.maximum(i, j)), 1)
        return m
    raise ValueError(f"pairing must be one of {PAIRINGS}, got {pairing!r}")


def generation_points(strategies, counts, rounds=200, pairing="round_robin", payoff=PAYOFF, noise=0.0, rng=None):
    """Total points scored by each strategy's agents in one generation."""
    rng = np.random.default_rng(rng)
    m = pair_counts(counts, pairing, rng)
    points = np.zeros(len(strategies))
    for i, j in zip(*np.nonzero(m)):
        s_i, s_j = play_totals(strategies[i], strategies[j], m[i, j], rounds, payoff, noise, rng)
        points[i] += s_i
        points[j] += s_j
    return points


def evolve(initial, generations=40, rounds=200, pairing="round_robin", payoff=PAYOFF, noise=0.0, rng=None):
    """Run the population game from ``initial``, a {strategy name or Strategy: number of agents} mapping.

    The population size stays fixed. Returns an ``Evolution``.
    """
    rng = np.random.default_rng(rng)
    strategies = get(initial)
    names = [s.name for s in strategies]
    counts = np.array(list(initial.values()), dtype=np.int64)
    total = int(counts.sum())
    population, points = [counts], []
    for _ in range(generations):
        p = generation_points(strategies, counts, rounds, pairing, payoff, noise, rng)
        counts = apportion(p, total)
        population.append(counts)
        points.append(p)
    population = pd.DataFrame(population, columns=names).rename_axis("generation")
    points = pd.DataFrame(points, columns=names).rename_axis("generation")
    return Evolution(population, points)


def replicator(matrix, shares=None, generations=40):
    """Discrete replicator dynamics for an infinite population.

    ``matrix`` is a player x opponent table of mean points per match (``tournament.score_matrix``). Each
    generation a strategy's share is multiplied by its average points against the current population, relative to
    the population average. Starts from equal shares unless ``shares`` is given. Returns shares per generation.
    """
    a = matrix.to_numpy(dtype=float)
    x = np.full(len(matrix), 1 / len(matrix)) if shares is None else np.asarray(shares, dtype=float)
    x = x / x.sum()
    rows = [x]
    for _ in range(generations):
        fitness = a @ x
        x = x * fitness / (x @ fitness)
        rows.append(x)
    return pd.DataFrame(rows, columns=matrix.index).rename_axis("generation")
