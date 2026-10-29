"""Negotiations in succession: coalitions settle among themselves first, then meet as single parties.

- ``succession``: play out one plan, given as nested lists of party names, with a table of every step
- ``outcomes``: the final position for many coalition structures at once (one row of block labels each)
- ``all_structures``: every way the parties can split into coalitions before the final negotiation
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import game, partitions


def succession(table, plan):
    """Play out ``plan``, nested lists of party names: each list is a negotiation among its entries, and a
    nested list is a coalition that settles first and enters the enclosing negotiation as one party. Parties not
    named in ``plan`` join the final negotiation on their own.

    Returns ``(outcome, steps)``: the final position and a DataFrame with one row per negotiation, in the order
    they settle (the final one last), giving its parties, position, combined influence and salience.
    """
    named = []
    steps = []

    def settle(node):
        if isinstance(node, str):
            named.append(node)
            return table.loc[[node]]
        group = pd.concat([settle(child) for child in node])
        merged = game.coalition(group)
        steps.append({"parties": list(group.index), **merged.iloc[0].to_dict()})
        return merged

    top = [settle(child) for child in plan]
    if len(set(named)) != len(named):
        raise ValueError("a party appears more than once in the plan")
    final = pd.concat(top + [table.drop(index=named)])
    merged = game.coalition(final, name="final")
    steps.append({"parties": list(final.index), **merged.iloc[0].to_dict()})
    return float(merged["position"].iloc[0]), pd.DataFrame(steps)


def outcomes(table, labels):
    """Final position for each coalition structure in ``labels`` (one row of block labels per structure, one
    column per party in ``table`` order): every block settles at its weighted average with ``coalition``'s
    influence and salience, then the blocks negotiate."""
    labels = np.asarray(labels)
    m, n = labels.shape
    p, i, s = (table[c].to_numpy() for c in game.COLUMNS)
    key = (np.arange(m)[:, None] * n + labels).ravel()  # one slot per (structure, block)

    def per_block(values):
        return np.bincount(key, weights=np.broadcast_to(values, (m, n)).ravel(), minlength=m * n)

    size = np.bincount(key, minlength=m * n)
    si = per_block(s * i)
    position = np.divide(per_block(s * i * p), si, out=np.zeros_like(si), where=size > 0)
    influence = per_block(i)
    top = np.zeros(m * n)
    np.maximum.at(top, key, np.broadcast_to(s, (m, n)).ravel())
    rest = np.divide(per_block(s) - top, size - 1, out=np.zeros_like(top), where=size > 1)
    salience = np.where(size > 1, (1 - game.OTHERS_WEIGHT) * top + game.OTHERS_WEIGHT * rest, top)
    w = (salience * influence).reshape(m, n)
    return (w * position.reshape(m, n)).sum(axis=1) / w.sum(axis=1)


@dataclass
class Structures:
    """Every coalition structure of a set of parties and the final position each one leads to."""
    names: list
    labels: np.ndarray    # (structures, parties) block labels, lexicographic order
    outcome: np.ndarray   # final position for each structure
    coalitions: np.ndarray  # number of blocks in each structure

    def blocks(self, k):
        """Structure ``k`` as a list of coalitions, each a list of party names."""
        return partitions.blocks(self.labels[k], self.names)

    def describe(self, k):
        """Structure ``k`` as text, e.g. ``"d1+me | legal | ..."``."""
        return " | ".join("+".join(b) for b in self.blocks(k))

    def together(self, a, b):
        """Mask of the structures where parties ``a`` and ``b`` are in the same coalition."""
        return self.labels[:, self.names.index(a)] == self.labels[:, self.names.index(b)]

    def pair_means(self):
        """Mean final position over the structures where each pair of parties is in the same coalition (the
        diagonal: where the party stands alone)."""
        n = len(self.names)
        out = np.empty((n, n))
        for x in range(n):
            same = self.labels == self.labels[:, [x]]
            alone = same.sum(axis=1) == 1
            out[x, x] = self.outcome[alone].mean()
            for y in range(x + 1, n):
                out[x, y] = out[y, x] = self.outcome[same[:, y]].mean()
        return pd.DataFrame(out, index=self.names, columns=self.names)


def all_structures(table):
    """Final position for every way the parties in ``table`` can split into coalitions (``bell(n)`` of them:
    678,570 for 11 parties)."""
    labels = partitions.growth_strings(len(table))
    return Structures(list(table.index), labels, outcomes(table, labels), labels.max(axis=1) + 1)
