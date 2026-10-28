import numpy as np
import pandas as pd
import pytest

from prisoners_dilemma import evolution, strategies, tournament

START_2008 = {n: 25 for n in ["T4TDefect", "Random", "T4T", "AvgT4T", "Defect", "Pred1", "T4TForgive"]}


@pytest.mark.parametrize("weights,total,expected", [([1, 1, 1], 10, [4, 3, 3]), ([0.5, 0.5], 3, [2, 1]),
                                                    ([0, 2, 1], 7, [0, 5, 2]), ([0, 0], 5, [0, 0])])
def test_apportion(weights, total, expected):
    assert list(evolution.apportion(weights, total)) == expected


def test_apportion_keeps_total():
    rng = np.random.default_rng(0)
    for _ in range(100):
        w = rng.random(7)
        assert evolution.apportion(w, 175).sum() == 175


def test_round_robin_pair_counts():
    m = evolution.pair_counts([3, 2])
    np.testing.assert_array_equal(m, [[3, 6], [0, 1]])
    assert m.sum() == 5 * 4 // 2


def test_random_pair_counts():
    m = evolution.pair_counts([3, 4], "random", rng=0)
    assert m.sum() == 3  # 7 agents: three matches, one agent sits out
    assert (np.tril(m, -1) == 0).all()
    with pytest.raises(ValueError):
        evolution.pair_counts([3, 4], "swiss")


def test_evolve_keeps_population_size():
    ev = evolution.evolve(START_2008, generations=6, rng=1)
    assert (ev.population.sum(axis=1) == 175).all()
    assert ev.shares.sum(axis=1).to_numpy() == pytest.approx(1.0)
    assert len(ev.points) == 6


def test_evolve_reproduces_2008_first_generations():
    # first two rows of resources/poprecord.csv (before the 2008 rounding drift set in)
    ev = evolution.evolve(START_2008, generations=2, rng=1)
    expected = pd.DataFrame([{"Random": 25, "T4TForgive": 28, "Defect": 16, "T4T": 28, "AvgT4T": 30,
                              "Pred1": 24, "T4TDefect": 24},
                             {"Random": 24, "T4TForgive": 31, "Defect": 10, "T4T": 30, "AvgT4T": 35,
                              "Pred1": 23, "T4TDefect": 22}])
    pd.testing.assert_frame_equal(ev.population.iloc[1:].reset_index(drop=True)[expected.columns], expected,
                                  check_dtype=False)


def test_evolve_accepts_strategy_objects():
    always_defect = strategies.Strategy("Mine", "always defects", lambda view, rng: view.all(False))
    ev = evolution.evolve({"T4T": 10, always_defect: 10}, generations=5, rng=0)
    assert list(ev.population.columns) == ["T4T", "Mine"]
    assert ev.population.iloc[-1]["T4T"] > 10


def test_random_pairing_runs():
    ev = evolution.evolve({"T4T": 5, "Defect": 6}, generations=3, pairing="random", rng=2)
    assert (ev.population.sum(axis=1) == 11).all()


def test_replicator():
    m = tournament.score_matrix(tournament.head_to_head(["T4T", "Defect", "Cooperate"]))
    shares = evolution.replicator(m, generations=200)
    assert shares.sum(axis=1).to_numpy() == pytest.approx(1.0)
    assert shares.iloc[-1]["Defect"] < 1e-3
    # without defectors around, T4T and Cooperate score the same, so their ratio freezes
    last = shares.iloc[-1]
    assert last["T4T"] > last["Cooperate"] > 0.1
