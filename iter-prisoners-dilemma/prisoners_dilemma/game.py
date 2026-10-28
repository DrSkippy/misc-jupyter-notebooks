"""Matches of the iterated prisoner's dilemma.

- ``Payoff``: the points for each pair of moves (default T=5, R=3, P=1, S=0, as in 2008)
- ``play``: many independent matches between two strategies at once, one array entry per match
- ``play_totals``: total points of each side over many matches
"""
from dataclasses import dataclass

import numpy as np

from .strategies import View


@dataclass(frozen=True)
class Payoff:
    temptation: float = 5  # I defect, you cooperate
    reward: float = 3  # both cooperate
    punishment: float = 1  # both defect
    sucker: float = 0  # I cooperate, you defect

    def __post_init__(self):
        t, r, p, s = self.temptation, self.reward, self.punishment, self.sucker
        if not (t > r > p > s and 2 * r > t + s):
            raise ValueError(f"not a prisoner's dilemma: need T > R > P > S and 2R > T + S, got {self}")

    def score(self, me, opp):
        """Points to the player who moved ``me`` against ``opp`` (bool arrays, True = cooperate)."""
        return np.where(me, np.where(opp, self.reward, self.sucker), np.where(opp, self.temptation, self.punishment))


PAYOFF = Payoff()


@dataclass(frozen=True)
class MatchResult:
    """Totals per match: points and fraction of moves that were cooperation, for players a and b."""
    score_a: np.ndarray
    score_b: np.ndarray
    coop_a: np.ndarray
    coop_b: np.ndarray


def play(a, b, n_matches=1, rounds=200, payoff=PAYOFF, noise=0.0, rng=None):
    """Play ``n_matches`` independent matches of ``rounds`` moves between strategies ``a`` and ``b``.

    With ``noise`` > 0 each intended move is flipped with that probability (a mistake); both players see the
    move actually made. Returns a ``MatchResult``.
    """
    rng = np.random.default_rng(rng)
    n = n_matches
    yes = np.ones(n, dtype=bool)
    last_a, last_b, prev_a, prev_b = yes, yes, yes, yes
    coops_a, coops_b = np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
    score_a, score_b = np.zeros(n), np.zeros(n)
    for t in range(rounds):
        move_a = a(View(t, last_a, last_b, prev_b, coops_b, t - coops_b), rng)
        move_b = b(View(t, last_b, last_a, prev_a, coops_a, t - coops_a), rng)
        if noise:
            move_a = move_a ^ (rng.random(n) < noise)
            move_b = move_b ^ (rng.random(n) < noise)
        score_a += payoff.score(move_a, move_b)
        score_b += payoff.score(move_b, move_a)
        coops_a += move_a
        coops_b += move_b
        prev_a, prev_b, last_a, last_b = last_a, last_b, move_a, move_b
    return MatchResult(score_a, score_b, coops_a / rounds, coops_b / rounds)


def play_totals(a, b, n_matches, rounds=200, payoff=PAYOFF, noise=0.0, rng=None):
    """Total points of a and b over ``n_matches`` matches. A pair with no randomness is played only once."""
    if n_matches == 0:
        return 0.0, 0.0
    if not (a.random or b.random or noise):
        r = play(a, b, 1, rounds, payoff, rng=rng)
        return float(r.score_a[0]) * n_matches, float(r.score_b[0]) * n_matches
    r = play(a, b, n_matches, rounds, payoff, noise, rng)
    return float(r.score_a.sum()), float(r.score_b.sum())
