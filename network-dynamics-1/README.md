# Random Network Growth: When Does Everything Connect?

Add random links to 1,000 unconnected nodes, 100 at a time, and watch the network fill in. Once the mean degree
(links per node × 2) passes 1, a giant component appears and within a few hundred more links holds most of the
nodes. This is the Erdős–Rényi random graph, the no-structure baseline for real networks. The notebook compares
the simulation with the theory: the giant component, Poisson degrees, isolated nodes, clustering, and the critical
point at mean degree 1, where the largest component grows as n^(2/3). A rich-get-richer network with the same
number of links shows what the random graph lacks: hubs.

Rewritten in 2026 from a set of Python 2 classes of the 2000s: a `NodeManager` singleton of `Node` objects, a
`NetBuilder` adding random links, and analyzers for degrees, components (`NodeTree`, breadth-first) and clustering
(`NodeNeighborhood`). Before the classes were removed, they were run under Python 3 on the same 5,000 links as the
rewrite: links, degree histogram, number and sizes of components and mean component size agree at all 50 steps,
and the mean clustering coefficient to 2e-17.

## Layout

| Path | Contents |
|---|---|
| `network_dynamics/graph.py` | random edge sequences (distinct pairs, in the order added), Barabási–Albert preferential attachment, degrees, connected components (scipy), clustering coefficients |
| `network_dynamics/growth.py` | grow a network `step` links at a time and measure it after each step; many seeds; degree counts per step |
| `network_dynamics/theory.py` | Erdős–Rényi expectations (giant-component fraction by Lambert W, Poisson degrees, isolated nodes, clustering) and the Barabási–Albert degree tail and degree by arrival order |
| `notebooks/random-network-growth.ipynb` | the 2000s run (1,000 nodes, 5,000 links) and 20 more, degrees, isolated nodes and clustering vs theory, networks of 100 to 100,000 nodes, the largest component at c = 1 up to a million nodes, the rich-get-richer comparison (about 20 s) |
| `tests/` | unit tests, including the small graphs and values of the 2000s tests and the 2000s code's results on a fixed run |

The notebook adds the project folder to `sys.path` itself, so nothing needs installing in the Jupyter image.

## Results

- The largest component follows the theory, S = 1 − e^(−cS) for mean degree c > 1. With 1,000 nodes it holds
  20% of the nodes at c = 1, 79% at c = 2 and all of them by c ≈ 10. The second largest component peaks near
  c = 1 at about 40 nodes.
- Degrees are Poisson: at c = 10 no node has more than 20 links. Random graphs have no hubs.
- Clustering is c / (n − 1), 1% at c = 10 with 1,000 nodes, far below that of social networks.
- The transition sharpens with size. At c = 1 the largest component is about n^(2/3) nodes (0.8 to 1.0 times it, from
  100 to 1,000,000 nodes), so its share of the network falls from 20% to 1%.
- With the same nodes and nearly the same links, rich get richer (each new node links to 5 earlier ones in
  proportion to their degree) gives a power-law tail: the largest degree is about 116 instead of 22, and the top
  1% of nodes hold 8.5% of the link ends instead of 1.9%. The hubs are the oldest nodes.
- So "winner takes all" fits the random graph only for components: one giant, chosen by chance, but no hubs.
