import numpy as np
import pytest

from network_dynamics import graph, growth


@pytest.fixture(scope="module")
def run():
    # the 2000s main.py run: 1000 nodes, 5000 links, measured every 100
    return growth.grow(1000, 5000, 100, rng=7)


def test_checkpoints(run):
    edges, table = run
    assert len(edges) == 5000
    assert table["links"].tolist() == list(range(100, 5001, 100))
    assert table["mean degree"].iloc[-1] == 10


def test_matches_2000s_code(run):
    # values of the 2000s NetAnalyzer / NodeTreeAnalyzer / NodeNeighborhoodAnalyzer (run under Python 3) on these
    # same edges, at 500, 1000, 2000 and 5000 links
    table = run[1].set_index("links")
    assert table.loc[[500, 1000, 2000, 5000], "components"].tolist() == [500, 168, 25, 1]
    assert table.loc[[500, 1000, 2000, 5000], "largest"].tolist() == [200, 789, 976, 1000]
    assert table.loc[2000, "mean size"] == 40
    assert table.loc[5000, "max degree"] == 20


def test_components_never_merge_back(run):
    table = run[1]
    assert (np.diff(table["components"]) <= 0).all()
    assert (np.diff(table["largest"]) >= 0).all()
    assert (table["second"] <= table["largest"]).all()


def test_measure_small_graph():
    edges = np.array([[0, 1], [0, 2], [0, 3], [3, 4], [3, 5], [1, 2]])
    row = growth.measure(7, edges)
    assert row["components"] == 2 and row["largest"] == 6 and row["second"] == 1
    assert row["mean size"] == 3.5 and row["isolated"] == 1
    assert row["mean degree"] == pytest.approx(12 / 7)
    assert row["clustering"] == pytest.approx(1 / 3)


def test_replay_given_edges(run):
    edges, table = run
    _, again = growth.grow(1000, 5000, 100, edges=edges)
    assert again.equals(table)


def test_uneven_last_step():
    _, table = growth.grow(100, 250, 100, rng=0)
    assert table["links"].tolist() == [100, 200, 250]


def test_degree_table(run):
    edges, _ = run
    d = growth.degree_table(1000, edges, 1000)
    assert d.index.tolist() == [1000, 2000, 3000, 4000, 5000]
    assert (d.sum(axis=1) == 1000).all()
    assert ((d * d.columns).sum(axis=1) == 2 * d.index).all()
    assert d.loc[5000].tolist() == np.bincount(graph.degrees(1000, edges)).tolist()


def test_grow_many():
    t = growth.grow_many(200, 400, 100, seeds=[0, 1, 2])
    assert len(t) == 12 and sorted(t["seed"].unique()) == [0, 1, 2]
