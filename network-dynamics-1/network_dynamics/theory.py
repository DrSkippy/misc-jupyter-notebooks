"""What to expect of a large random graph with mean degree c = 2m / n (Erdős–Rényi).

- ``giant_fraction``: share of the nodes in the largest component, the root S of S = 1 - exp(-c S) (0 for c <= 1)
- ``degree_pmf``: Poisson(c), the limiting degree distribution
- ``clustering``: c / (n - 1), the chance that two neighbors of a node are linked
- ``isolated_fraction``: exp(-c), the share of nodes with no links
- ``degree_ccdf``: P(degree >= k) for the random graph (Poisson)
- ``preferential_ccdf``: P(degree >= k) = m (m + 1) / (k (k + 1)) for a Barabási–Albert network with m links per
  new node, a power-law tail
- ``preferential_degree``: expected degree of the node that arrived i-th, m sqrt(n / i)
"""
import numpy as np
from scipy import special, stats


def giant_fraction(c):
    """Fraction of nodes in the giant component for mean degree ``c`` (scalar or array)."""
    c = np.asarray(c, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = 1 + special.lambertw(-c * np.exp(-c)).real / c  # principal branch: the nonzero root for c > 1
    return np.where(c > 1, s, 0.0)


def degree_pmf(k, c):
    """Probability that a node has degree ``k`` when the mean degree is ``c``."""
    return stats.poisson.pmf(k, c)


def clustering(c, n):
    """Expected clustering coefficient: c / (n - 1)."""
    return np.asarray(c, dtype=float) / (n - 1)


def isolated_fraction(c):
    """Expected fraction of nodes with no links: exp(-c)."""
    return np.exp(-np.asarray(c, dtype=float))


def degree_ccdf(k, c):
    """Probability that a node has degree ``k`` or more in a random graph of mean degree ``c``."""
    return stats.poisson.sf(np.asarray(k) - 1, c)


def preferential_ccdf(k, m):
    """Probability that a node has degree ``k`` or more in a large Barabási–Albert network with ``m`` links per
    new node: m (m + 1) / (k (k + 1)) for k >= m, falling as k^-2."""
    k = np.asarray(k, dtype=float)
    return np.where(k <= m, 1.0, m * (m + 1) / (k * (k + 1)))


def preferential_degree(i, n, m):
    """Expected final degree of the node that arrived at step ``i`` (counting from 1) out of ``n``: m sqrt(n / i)."""
    return m * np.sqrt(n / np.asarray(i, dtype=float))
