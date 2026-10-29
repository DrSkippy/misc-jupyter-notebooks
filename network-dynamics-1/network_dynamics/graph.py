"""Random graphs as edge arrays, and the measurements taken on them.

A graph is ``n`` nodes numbered 0..n-1 and an ``(m, 2)`` int array of undirected edges, with no self-loops and
no repeated pairs.

- ``random_edges``: m distinct random edges in the order they are added (the G(n, m) random graph, grown)
- ``preferential_edges``: nodes arriving one at a time, each linking to existing nodes in proportion to their
  degree (the Barabási–Albert rich-get-richer network)
- ``degrees``: links per node
- ``component_labels`` / ``component_sizes``: connected groups of nodes
- ``clustering``: each node's clustering coefficient (fraction of its neighbor pairs that are linked)
"""
import numpy as np
from scipy import sparse
from scipy.sparse import csgraph


def random_edges(n, m, rng=None):
    """``m`` distinct edges between ``n`` nodes, each pair equally likely, in the order they are added: every new
    edge joins two different nodes drawn uniformly at random, redrawn if the pair is already linked."""
    if m > n * (n - 1) // 2:
        raise ValueError(f"{n} nodes have at most {n * (n - 1) // 2} distinct edges")
    rng = np.random.default_rng(rng)
    keys = np.empty(0, dtype=np.int64)
    while len(keys) < m:
        a, b = rng.integers(0, n, size=(2, 2 * (m - len(keys)) + 16))
        draw = np.minimum(a, b) * n + np.maximum(a, b)
        draw = np.concatenate([keys, draw[a != b]])
        _, first = np.unique(draw, return_index=True)
        keys = draw[np.sort(first)][:m]  # first time each pair is drawn, in draw order
    return np.column_stack([keys // n, keys % n])


def preferential_edges(n, k, rng=None):
    """Edges of a Barabási–Albert network: ``k + 1`` founding nodes all linked to each other, then nodes
    k + 1, ..., n - 1 arrive in order and each links to ``k`` distinct earlier nodes, picking each with probability
    proportional to its degree. Returns ``k (k + 1) / 2 + (n - k - 1) k`` edges in the order added, as
    (earlier node, new node)."""
    if n < k + 1:
        raise ValueError(f"need at least {k + 1} nodes for {k} links per new node")
    rng = np.random.default_rng(rng)
    seed = [(i, j) for i in range(k + 1) for j in range(i + 1, k + 1)]
    edges = np.empty((len(seed) + (n - k - 1) * k, 2), dtype=np.int64)
    edges[:len(seed)] = seed
    ends = np.empty(2 * len(edges), dtype=np.int64)  # every edge's two endpoints: a node appears once per link
    ends[:2 * len(seed)] = edges[:len(seed)].ravel()
    row, filled = len(seed), 2 * len(seed)
    for new in range(k + 1, n):
        targets = set()
        while len(targets) < k:  # a uniform draw from ends picks a node with probability proportional to degree
            targets.update(ends[rng.integers(0, filled, k - len(targets))].tolist())
        targets = sorted(targets)
        edges[row:row + k] = [(t, new) for t in targets]
        ends[filled:filled + 2 * k] = edges[row:row + k].ravel()
        row, filled = row + k, filled + 2 * k
    return edges


def adjacency(n, edges):
    """Symmetric sparse 0/1 adjacency matrix."""
    edges = np.asarray(edges).reshape(-1, 2)
    i = np.concatenate([edges[:, 0], edges[:, 1]])
    j = np.concatenate([edges[:, 1], edges[:, 0]])
    return sparse.csr_matrix((np.ones(len(i), dtype=np.int64), (i, j)), shape=(n, n))


def degrees(n, edges):
    """Number of links at each node."""
    return np.bincount(np.asarray(edges).ravel(), minlength=n)


def component_labels(n, edges):
    """Connected-component label of each node (labels 0..k-1)."""
    return csgraph.connected_components(adjacency(n, edges), directed=False)[1]


def component_sizes(n, edges):
    """Sizes of the connected components, largest first (isolated nodes are components of size 1)."""
    return np.sort(np.bincount(component_labels(n, edges)))[::-1]


def clustering(n, edges):
    """Each node's clustering coefficient: links among its neighbors / possible links among them (0 for nodes
    with fewer than two neighbors)."""
    a = adjacency(n, edges)
    triangles = np.asarray((a @ a).multiply(a).sum(axis=1)).ravel() / 2
    k = degrees(n, edges)
    pairs = k * (k - 1) / 2
    return np.divide(triangles, pairs, out=np.zeros(n), where=pairs > 0)
