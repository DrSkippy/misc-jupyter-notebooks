"""Round-robin comparison of strategies.

- ``head_to_head``: every strategy plays every strategy (itself included) many times; one row per ordered pair
  with mean points, win/loss/tie fractions and cooperation rate
- ``score_matrix``: the mean points as a player x opponent table
- ``ranking``: strategies ordered by their average points over all opponents
"""
import numpy as np
import pandas as pd

from .game import PAYOFF, play
from .strategies import get


def head_to_head(strategies, rounds=200, repeats=200, payoff=PAYOFF, noise=0.0, rng=None):
    """``repeats`` matches of ``rounds`` moves for every pair of ``strategies`` (names or Strategy objects).

    Returns a DataFrame with one row per (player, opponent): mean ``score`` per match, ``win``/``loss``/``tie``
    fractions of matches, and ``coop``, the player's fraction of cooperative moves.
    """
    rng = np.random.default_rng(rng)
    strategies = get(strategies)
    rows = []
    for i, a in enumerate(strategies):
        for b in strategies[i:]:
            n = repeats if (a.random or b.random or noise) else 1
            r = play(a, b, n, rounds, payoff, noise, rng)
            sides = [(a, b, r.score_a, r.score_b, r.coop_a), (b, a, r.score_b, r.score_a, r.coop_b)]
            if a is b:  # both sides are the same strategy: pool them into one row
                sides = [(a, a, np.concatenate([r.score_a, r.score_b]), np.concatenate([r.score_b, r.score_a]),
                          np.concatenate([r.coop_a, r.coop_b]))]
            for me, opp, s_me, s_opp, coop in sides:
                rows.append({"player": me.name, "opponent": opp.name, "score": s_me.mean(), "sd": s_me.std(),
                             "win": (s_me > s_opp).mean(), "loss": (s_me < s_opp).mean(),
                             "tie": (s_me == s_opp).mean(), "coop": coop.mean()})
    return pd.DataFrame(rows)


def score_matrix(results, value="score"):
    """Player x opponent table of ``value`` from ``head_to_head`` results, rows and columns in input order."""
    order = list(dict.fromkeys(results["player"]))
    return results.pivot(index="player", columns="opponent", values=value).loc[order, order]


def ranking(results, self_play=True):
    """Average points per match over all opponents, best first, with each strategy's win fraction and
    cooperation rate."""
    if not self_play:
        results = results[results["player"] != results["opponent"]]
    table = results.groupby("player")[["score", "win", "loss", "tie", "coop"]].mean()
    return table.sort_values("score", ascending=False)
