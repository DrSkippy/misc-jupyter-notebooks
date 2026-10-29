import numpy as np
import pytest

from network_dynamics import graph

# the small graph of the 2000s tests: 1-2, 1-3, 1-4, 4-5, 4-6, 2-3, and node 7 alone (numbered from 0 here)
EDGES = np.array([[0, 1], [0, 2], [0, 3], [3, 4], [3, 5], [1, 2]])


def test_random_edges_distinct_and_valid():
    e = graph.random_edges(50, 600, rng=0)
    assert e.shape == (600, 2)
    assert (e[:, 0] < e[:, 1]).all() and e.min() >= 0 and e.max() < 50
    assert len(np.unique(e, axis=0)) == 600


def test_random_edges_prefix_is_reproducible():
    assert (graph.random_edges(100, 300, rng=3) == graph.random_edges(100, 300, rng=3)).all()


def test_random_edges_complete_graph():
    e = graph.random_edges(6, 15, rng=1)
    assert len(np.unique(e, axis=0)) == 15
    with pytest.raises(ValueError):
        graph.random_edges(6, 16)


def test_random_edges_uniform_over_pairs():
    # first edge of many short sequences: every one of the 10 pairs of 5 nodes about equally often
    firsts = [tuple(graph.random_edges(5, 1, rng=s)[0]) for s in range(5000)]
    counts = np.unique(firsts, axis=0, return_counts=True)[1]
    assert len(counts) == 10 and counts.min() > 400 and counts.max() < 600


def test_degrees():
    assert graph.degrees(7, EDGES).tolist() == [3, 2, 2, 3, 1, 1, 0]


def test_components():
    # 2000s nodeTreeAnalyzer test: groups of 6 and 1
    assert graph.component_sizes(7, EDGES).tolist() == [6, 1]
    labels = graph.component_labels(7, EDGES)
    assert len(set(labels[:6])) == 1 and labels[6] != labels[0]


def test_clustering():
    # 2000s nodeNeighborhoodAnalyzer test: [1/3, 1, 1, 0, 0, 0, 0], mean 1/3
    cc = graph.clustering(7, EDGES)
    assert cc == pytest.approx([1 / 3, 1, 1, 0, 0, 0, 0])
    assert cc.mean() == pytest.approx(1 / 3)


def test_clustering_complete_graph():
    e = np.array([(i, j) for i in range(5) for j in range(i + 1, 5)])
    assert graph.clustering(5, e) == pytest.approx(np.ones(5))


def test_empty_graph():
    e = np.zeros((0, 2), dtype=int)
    assert graph.component_sizes(4, e).tolist() == [1, 1, 1, 1]
    assert graph.clustering(4, e).tolist() == [0, 0, 0, 0]


def test_preferential_edges_shape():
    e = graph.preferential_edges(1000, 5, rng=0)
    assert len(e) == 15 + 994 * 5
    assert (e[:, 0] < e[:, 1]).all() and len(np.unique(e, axis=0)) == len(e)
    assert graph.component_sizes(1000, e).tolist() == [1000]  # every new node links to the network
    k = graph.degrees(1000, e)
    assert k.min() == 5


def test_preferential_edges_each_new_node_links_k_earlier_nodes():
    e = graph.preferential_edges(200, 3, rng=1)
    later = e[6:]
    assert (np.bincount(later[:, 1], minlength=200)[4:] == 3).all()
    assert (later[:, 0] < later[:, 1]).all()


def test_preferential_edges_rich_get_richer():
    # the oldest nodes collect far more links than the youngest, and the largest degree is far above a random
    # graph's with the same number of links
    k = np.mean([graph.degrees(2000, graph.preferential_edges(2000, 4, rng=s)) for s in range(5)], axis=0)
    assert k[:20].mean() > 5 * k[-1000:].mean()
    assert k.max() > 3 * graph.degrees(2000, graph.random_edges(2000, 4 * 2000, rng=0)).max()


def test_preferential_edges_needs_founders():
    with pytest.raises(ValueError):
        graph.preferential_edges(4, 5)
