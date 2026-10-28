import numpy as np
import pytest

from prisoners_dilemma import game, strategies

DETERMINISTIC = [s.name for s in strategies.STRATEGIES.values() if not s.random]


def legacy_move(name, opp_history):
    """The 2008 agents' rules, one move at a time from the opponent's history of 'c'/'s'."""
    if name == "Defect":
        return "s"
    if not opp_history:
        return "c"
    c, s = opp_history.count("c"), opp_history.count("s")
    if name == "T4T":
        return opp_history[-1]
    if name == "AvgT4T":
        return "c" if c > s else "s"
    if name == "Pred1":
        return "s" if (c > 1.1 * s) or (c < 0.4 * s) else "c"
    raise ValueError(name)


def legacy_match(n1, n2, rounds=200):
    h1, h2 = [], []
    for _ in range(rounds):
        m1, m2 = legacy_move(n1, h2), legacy_move(n2, h1)
        h1.append(m1)
        h2.append(m2)
    points = {("c", "c"): (3, 3), ("c", "s"): (0, 5), ("s", "c"): (5, 0), ("s", "s"): (1, 1)}
    s1 = sum(points[m][0] for m in zip(h1, h2))
    s2 = sum(points[m][1] for m in zip(h1, h2))
    return s1, s2


LEGACY = ["T4T", "AvgT4T", "Pred1", "Defect"]


@pytest.mark.parametrize("a", LEGACY)
@pytest.mark.parametrize("b", LEGACY)
def test_deterministic_strategies_match_2008_rules(a, b):
    r = game.play(strategies.STRATEGIES[a], strategies.STRATEGIES[b])
    assert (r.score_a[0], r.score_b[0]) == legacy_match(a, b)


def test_get_rejects_unknown_names():
    with pytest.raises(KeyError, match="Nice"):
        strategies.get(["T4T", "Nice"])
    t4t = strategies.STRATEGIES["T4T"]
    assert strategies.get([t4t, "Defect"])[0] is t4t


@pytest.mark.parametrize("name", DETERMINISTIC)
def test_deterministic_strategies_ignore_rng(name):
    s = strategies.STRATEGIES[name]
    r1 = game.play(s, strategies.STRATEGIES["Random"], 5, rng=1)
    r2 = game.play(s, strategies.STRATEGIES["Random"], 5, rng=1)
    np.testing.assert_array_equal(r1.score_a, r2.score_a)


def test_random_strategy_rates():
    S = strategies.STRATEGIES
    assert game.play(S["Random"], S["Cooperate"], 2000, rng=0).coop_a.mean() == pytest.approx(0.5, abs=0.01)
    # against a cooperator, T4TDefect defects on about 4% of moves after the first
    assert game.play(S["T4TDefect"], S["Cooperate"], 2000, rng=0).coop_a.mean() == pytest.approx(
        0.005 + 0.995 * 0.96, abs=0.005)
    # against a defector, T4TForgive cooperates on the first move and about 3% of the rest
    assert game.play(S["T4TForgive"], S["Defect"], 2000, rng=0).coop_a.mean() == pytest.approx(
        0.005 + 0.995 * 0.03, abs=0.005)


def test_classics():
    S = strategies.STRATEGIES
    # Grudger cooperates on the first move, then never forgives the defection
    assert game.play(S["Grudger"], S["SuspiciousT4T"], rounds=10).coop_a[0] == 0.1
    # two tit for tats with one suspicious alternate forever; TF2T forgives the first defection
    assert game.play(S["T4T"], S["SuspiciousT4T"], rounds=10).coop_a[0] == 0.5
    assert game.play(S["TF2T"], S["SuspiciousT4T"], rounds=10).coop_b[0] == 0.9
    # Pavlov cooperates with itself, and defects every other move against a defector
    assert game.play(S["Pavlov"], S["Pavlov"], rounds=10).coop_a[0] == 1
    assert game.play(S["Pavlov"], S["Defect"], rounds=10).coop_a[0] == 0.5
