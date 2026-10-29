from pathlib import Path

import numpy as np
import pytest

from negotiations import partitions

EXAMPLE = Path(__file__).resolve().parent.parent / "resources" / "partitions_example.txt"


@pytest.mark.parametrize("n", range(0, 12))
def test_count_is_bell_number(n):
    labels = partitions.growth_strings(n)
    assert len(labels) == partitions.bell(n)
    assert len(np.unique(labels, axis=0)) == len(labels)


def test_bell_numbers():
    assert [partitions.bell(n) for n in range(12)] == [1, 1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975, 678570]


def test_restricted_growth():
    labels = partitions.growth_strings(7).astype(int)
    assert (labels[:, 0] == 0).all()
    prev_max = np.maximum.accumulate(labels, axis=1)[:, :-1]
    assert (labels[:, 1:] <= prev_max + 1).all()


def test_lexicographic_order():
    labels = partitions.growth_strings(6)
    keys = [tuple(r) for r in labels]
    assert keys == sorted(keys)


def test_matches_2009_output():
    # resources/partitions_example.txt: the 2009 Partitioner.py (Knuth's Algorithm H) on a, b, c, d
    lines = EXAMPLE.read_text().strip().splitlines()[3:]
    names = ["a", "b", "c", "d"]
    assert [str(partitions.blocks(r, names)) for r in partitions.growth_strings(4)] == lines
