"""Strategies for the iterated prisoner's dilemma.

Rewrite of the 2008 Python 2 ``agent_*.py`` classes. A move is a bool: True = cooperate (stay mum),
False = defect (snitch). Every strategy plays many matches at once: it gets a ``View`` of the game so far, with
one entry per match, and returns an array of moves.

- ``ORIGINAL``: the seven strategies of the 2008 tournament, with the same rules and probabilities
- ``CLASSICS``: well-known strategies added for comparison
- ``STRATEGIES``: all of them by name; ``get`` looks names up
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np

COOPERATE, DEFECT = True, False


@dataclass(frozen=True)
class View:
    """What one player knows before move ``t`` (0-based) of each of ``n`` simultaneous matches.

    ``my_last``/``opp_last``/``opp_last2`` are the moves one and two turns back (all True before they exist);
    ``opp_coops``/``opp_defects`` count the opponent's moves so far.
    """
    t: int
    my_last: np.ndarray
    opp_last: np.ndarray
    opp_last2: np.ndarray
    opp_coops: np.ndarray
    opp_defects: np.ndarray

    @property
    def n(self):
        return self.opp_last.size

    def all(self, move):
        return np.full(self.n, move)


@dataclass(frozen=True)
class Strategy:
    name: str
    description: str
    move: Callable[[View, np.random.Generator], np.ndarray]
    random: bool = False

    def __call__(self, view, rng):
        return self.move(view, rng)

    def __repr__(self):
        return f"Strategy({self.name!r})"


def _first(view, rule, first=COOPERATE):
    """``first`` on move 0, otherwise ``rule()``."""
    return view.all(first) if view.t == 0 else rule()


def _random(view, rng):
    return rng.random(view.n) > 0.5


def _t4t(view, rng):
    return _first(view, lambda: view.opp_last.copy())


def _avg_t4t(view, rng):
    # a tie in the opponent's record counts as defection, as in 2008
    return _first(view, lambda: view.opp_coops > view.opp_defects)


def _pred1(view, rng):
    c, s = view.opp_coops, view.opp_defects
    return _first(view, lambda: ~((c > 1.1 * s) | (c < 0.4 * s)))


def _defect(view, rng):
    return view.all(DEFECT)


def _t4t_defect(view, rng):
    return _first(view, lambda: view.opp_last & (rng.random(view.n) <= 0.96))


def _t4t_forgive(view, rng):
    return _first(view, lambda: view.opp_last | (rng.random(view.n) > 0.97))


def _cooperate(view, rng):
    return view.all(COOPERATE)


def _grudger(view, rng):
    return view.opp_defects == 0


def _pavlov(view, rng):
    # win-stay, lose-shift: cooperate after both made the same move
    return _first(view, lambda: view.my_last == view.opp_last)


def _tf2t(view, rng):
    return view.opp_last | view.opp_last2


def _suspicious_t4t(view, rng):
    return _first(view, lambda: view.opp_last.copy(), first=DEFECT)


ORIGINAL = [
    Strategy("Random", "flips a fair coin every move", _random, random=True),
    Strategy("T4T", "tit for tat: cooperates first, then copies the opponent's last move", _t4t),
    Strategy("AvgT4T", "cooperates first, then cooperates while the opponent has cooperated more often than "
             "defected", _avg_t4t),
    Strategy("Pred1", "cooperates first, then exploits an opponent who cooperates more than 1.1x as often as "
             "they defect, and retaliates against one who cooperates less than 0.4x as often", _pred1),
    Strategy("Defect", "always defects", _defect),
    Strategy("T4TDefect", "tit for tat, but defects at random 4% of the time", _t4t_defect, random=True),
    Strategy("T4TForgive", "tit for tat, but forgives a defection 3% of the time", _t4t_forgive, random=True),
]

CLASSICS = [
    Strategy("Cooperate", "always cooperates", _cooperate),
    Strategy("Grudger", "cooperates until the opponent defects once, then always defects", _grudger),
    Strategy("Pavlov", "win-stay, lose-shift: cooperates first, then cooperates if both players made the same "
             "move last turn", _pavlov),
    Strategy("TF2T", "tit for two tats: defects only after two defections in a row", _tf2t),
    Strategy("SuspiciousT4T", "tit for tat, but defects first", _suspicious_t4t),
]

STRATEGIES = {s.name: s for s in ORIGINAL + CLASSICS}


def get(names):
    """Strategies for a list of names (or Strategy objects, passed through). Raises KeyError on unknown names."""
    out = []
    for n in names:
        if isinstance(n, Strategy):
            out.append(n)
        elif n in STRATEGIES:
            out.append(STRATEGIES[n])
        else:
            raise KeyError(f"unknown strategy {n!r}; choose from {sorted(STRATEGIES)}")
    return out
