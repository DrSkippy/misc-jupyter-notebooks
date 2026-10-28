import pytest

from prisoners_dilemma import strategies, tournament

NAMES = [s.name for s in strategies.ORIGINAL]


@pytest.fixture(scope="module")
def results():
    return tournament.head_to_head(NAMES, repeats=100, rng=0)


def test_one_row_per_ordered_pair(results):
    assert len(results) == len(NAMES) ** 2
    assert not results.duplicated(["player", "opponent"]).any()
    assert results[["win", "loss", "tie"]].sum(axis=1).to_numpy() == pytest.approx(1.0)


def test_score_matrix(results):
    m = tournament.score_matrix(results)
    assert list(m.index) == NAMES and list(m.columns) == NAMES
    assert m.loc["T4T", "Defect"] == 199 and m.loc["Defect", "T4T"] == 204
    assert m.loc["T4T", "AvgT4T"] == 600


def test_wins_and_losses_mirror(results):
    w = tournament.score_matrix(results, "win")
    l = tournament.score_matrix(results, "loss")
    assert (w == l.T).all().all()


def test_defect_never_loses_but_ranks_last(results):
    rank = tournament.ranking(results)
    assert rank.index[-1] == "Defect"
    assert rank.loc["Defect", "loss"] == 0
    assert rank.index[0] == "AvgT4T"


def test_reproducible():
    a = tournament.head_to_head(["Random", "T4TDefect"], repeats=20, rng=5)
    b = tournament.head_to_head(["Random", "T4TDefect"], repeats=20, rng=5)
    assert a.equals(b)
