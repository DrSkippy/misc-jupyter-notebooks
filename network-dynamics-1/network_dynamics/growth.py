"""Growing a random network a batch of links at a time, measuring it after each batch.

- ``measure``: one row of statistics for a graph
- ``grow``: add ``m`` random links ``step`` at a time and measure after each step (the 2000s ``NetBuilder``)
- ``grow_many``: the same for several seeds, one table
- ``degree_table``: how many nodes have each degree, after each step
"""
import numpy as np
import pandas as pd

from . import graph


def measure(n, edges):
    """Statistics of a graph: links, mean degree, number of components, largest component (nodes and fraction),
    second largest component, mean component size, isolated nodes, largest degree, mean clustering coefficient."""
    sizes = graph.component_sizes(n, edges)
    k = graph.degrees(n, edges)
    return {"links": len(edges), "mean degree": 2 * len(edges) / n, "components": len(sizes),
            "largest": int(sizes[0]), "largest fraction": sizes[0] / n,
            "second": int(sizes[1]) if len(sizes) > 1 else 0, "mean size": n / len(sizes),
            "isolated": int((k == 0).sum()), "max degree": int(k.max()),
            "clustering": graph.clustering(n, edges).mean()}


def grow(n, m, step, rng=None, edges=None):
    """Add ``m`` random links to ``n`` nodes, ``step`` at a time, measuring after each step. Returns
    ``(edges, table)``: the links in the order added and a DataFrame with one row per step. Pass ``edges`` to
    replay a given sequence instead of drawing one."""
    if edges is None:
        edges = graph.random_edges(n, m, rng)
    checkpoints = list(range(step, m, step)) + [m]
    table = pd.DataFrame([measure(n, edges[:t]) for t in checkpoints])
    return edges, table


def grow_many(n, m, step, seeds):
    """``grow`` once per seed; one table with a ``seed`` column."""
    return pd.concat([grow(n, m, step, rng=s)[1].assign(seed=s) for s in seeds], ignore_index=True)


def degree_table(n, edges, step):
    """Number of nodes with each degree (columns) after every ``step`` links (rows, indexed by links)."""
    checkpoints = list(range(step, len(edges), step)) + [len(edges)]
    counts = [np.bincount(graph.degrees(n, edges[:t])) for t in checkpoints]
    width = max(len(c) for c in counts)
    data = np.array([np.pad(c, (0, width - len(c))) for c in counts])
    return pd.DataFrame(data, index=pd.Index(checkpoints, name="links"), columns=range(width))
