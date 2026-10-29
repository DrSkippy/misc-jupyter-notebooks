import numpy as np
import pytest

from negotiations import example, game, model, partitions


@pytest.fixture(scope="module")
def table():
    return game.parties(example.SPREADSHEET)


@pytest.fixture(scope="module")
def structures(table):
    return model.all_structures(table)


@pytest.fixture(scope="module")
def structures_2009():
    # the 2009 code listed partitions over the party names sorted alphabetically
    return model.all_structures(game.parties(example.SCRIPT_2009).sort_index())


@pytest.mark.parametrize("k, expected", [(0, 76.36764705882352), (1, 76.53601466770952), (1000, 72.0059633614422),
                                         (47613, 66.27526816857457), (2087, 83.2214739551376),
                                         (115974, 76.36764705882352)])
def test_matches_2009_model(structures_2009, k, expected):
    # outcomes of the 2009 MultiPlayerNegotiationsModel.modelGames, run under Python 3
    assert structures_2009.outcome[k] == pytest.approx(expected, abs=1e-9)


def test_2009_extremes(structures_2009):
    s = structures_2009
    assert s.outcome.argmin() == 47613 and s.outcome.argmax() == 2087
    assert s.describe(47613) == "adv | ctcust+ctsal+d2+inv | d1+eng+me | er+legal"


def test_spreadsheet_extremes(structures):
    assert len(structures.outcome) == 678570
    assert structures.outcome.min() == pytest.approx(65.94643352329572)
    assert structures.outcome.max() == pytest.approx(83.82512395995408)


def test_grand_coalition_and_all_alone_agree(table, structures):
    # one coalition of everybody, or nobody in a coalition: both are the single negotiation
    assert structures.coalitions[0] == 1 and structures.coalitions[-1] == len(table)
    assert structures.outcome[0] == pytest.approx(game.weighted_average(table))
    assert structures.outcome[-1] == pytest.approx(game.weighted_average(table))


def test_succession_agrees_with_outcomes(table, structures):
    rng = np.random.default_rng(1)
    for k in rng.integers(0, len(structures.outcome), 20):
        plan = [b for b in structures.blocks(k) if len(b) > 1]
        outcome, steps = model.succession(table, plan)
        assert outcome == pytest.approx(structures.outcome[k])
        assert len(steps) == len(plan) + 1


def test_succession_nested(table):
    outcome, steps = model.succession(table, [[["d1", "me"], "engineer"], ["d2", "potential investor"]])
    assert steps["parties"].tolist()[:3] == [["d1", "me"], ["d1+me", "engineer"], ["d2", "potential investor"]]
    final = steps.iloc[-1]
    assert sorted(final["parties"]) == sorted(["d1+me+engineer", "d2+potential investor", "customer rep",
                                               "sales staff", "erin", "Customer 1", "advisor", "legal"])
    assert final["influence"] == table["influence"].sum()
    assert outcome == final["position"]


def test_succession_rejects_repeats(table):
    with pytest.raises(ValueError):
        model.succession(table, [["d1", "me"], ["me", "legal"]])


def test_pair_means(structures):
    m = structures.pair_means()
    assert m.loc["d1", "me"] == pytest.approx(structures.outcome[structures.together("d1", "me")].mean())
    assert (m == m.T).all().all()
    alone = (structures.labels == structures.labels[:, [0]]).sum(axis=1) == 1
    assert m.iloc[0, 0] == pytest.approx(structures.outcome[alone].mean())


def test_outcomes_small_case():
    t = game.parties({"a": [0, 1, 100], "b": [100, 1, 50], "c": [50, 2, 10]})
    labels = partitions.growth_strings(3)
    got = model.outcomes(t, labels)
    for row, value in zip(labels, got):
        plan = [b for b in partitions.blocks(row, list(t.index)) if len(b) > 1]
        assert value == pytest.approx(model.succession(t, plan)[0])
