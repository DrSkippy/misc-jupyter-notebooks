"""Every way of splitting a set of parties into coalitions (set partitions).

A partition of n items is written as a restricted growth string: item i gets a block label, the first item is in
block 0, and each later item either joins an existing block or opens the next one (label at most 1 + the largest
label before it). Listing the strings in lexicographic order gives the same order as Knuth's Algorithm H
(TAOCP 7.2.1.5), which the 2009 code used.

- ``growth_strings``: all partitions of n items as an array of block labels, one row per partition
- ``bell``: the number of partitions of n items
- ``blocks``: one partition as lists of names
"""
import numpy as np


def growth_strings(n):
    """All ``bell(n)`` partitions of n items, as an int8 array of block labels with one row per partition,
    in lexicographic order."""
    if n == 0:
        return np.zeros((1, 0), dtype=np.int8)
    labels = np.zeros((1, 1), dtype=np.int8)
    top = np.zeros(1, dtype=np.int8)  # largest label so far in each row
    for _ in range(1, n):
        choices = top.astype(np.int64) + 2  # join one of the blocks 0..top, or open block top + 1
        rows = np.repeat(np.arange(len(labels)), choices)
        start = np.repeat(np.cumsum(choices) - choices, choices)
        new = (np.arange(len(rows)) - start).astype(np.int8)
        labels = np.column_stack([labels[rows], new])
        top = np.maximum(top[rows], new)
    return labels


def bell(n):
    """Number of partitions of n items, from the Bell triangle."""
    row = [1]
    for _ in range(n):
        nxt = [row[-1]]
        for x in row:
            nxt.append(nxt[-1] + x)
        row = nxt
    return row[0]


def blocks(labels, names):
    """One partition (a row of block labels) as a list of blocks, each a list of names, in label order."""
    out = {}
    for label, name in zip(labels, names):
        out.setdefault(int(label), []).append(name)
    return list(out.values())
