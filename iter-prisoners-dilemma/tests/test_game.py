import numpy as np
import pytest

from prisoners_dilemma import game
from prisoners_dilemma.strategies import STRATEGIES as S


def test_payoff_table():
    p = game.PAYOFF
    me = np.array([True, True, False, False])
    opp = np.array([True, False, True, False])
    np.testing.assert_array_equal(p.score(me, opp), [3, 0, 5, 1])


def test_payoff_must_be_a_dilemma():
    with pytest.raises(ValueError):
        game.Payoff(temptation=3, reward=5)
    with pytest.raises(ValueError):
        game.Payoff(temptation=10)  # alternating C/D would beat mutual cooperation


def test_2008_match_score_shares():
    # "Match Scores" in resources/game.scores: share of the points won by the first player
    for a, b, share in [("Defect", "T4T", 0.506), ("Pred1", "T4T", 0.503), ("Pred1", "AvgT4T", 0.522),
                        ("Defect", "Pred1", 0.506), ("AvgT4T", "T4T", 0.500)]:
        r = game.play(S[a], S[b])
        assert r.score_a[0] / (r.score_a[0] + r.score_b[0]) == pytest.approx(share, abs=5e-4)


def test_known_scores():
    r = game.play(S["T4T"], S["Defect"], 3, rounds=200)
    np.testing.assert_array_equal(r.score_a, 199)
    np.testing.assert_array_equal(r.score_b, 204)
    assert game.play(S["Cooperate"], S["Cooperate"], rounds=7).score_a[0] == 21


def test_noise_flips_moves():
    r = game.play(S["Cooperate"], S["Defect"], 1000, rounds=100, noise=0.1, rng=0)
    assert r.coop_a.mean() == pytest.approx(0.9, abs=0.01)
    assert r.coop_b.mean() == pytest.approx(0.1, abs=0.01)


def test_noise_breaks_t4t_mirror():
    clean = game.play(S["T4T"], S["T4T"], 200, rng=0)
    noisy = game.play(S["T4T"], S["T4T"], 200, noise=0.01, rng=0)
    assert clean.score_a.mean() == 600
    assert noisy.score_a.mean() < 500


def test_play_totals_shortcut_matches_full_play():
    for a, b in [("T4T", "Pred1"), ("Random", "AvgT4T")]:
        n = 50
        total = game.play_totals(S[a], S[b], n, rng=3)
        full = game.play(S[a], S[b], n, rng=3)
        assert total == pytest.approx((full.score_a.sum(), full.score_b.sum()))
    assert game.play_totals(S["T4T"], S["T4T"], 0) == (0.0, 0.0)
