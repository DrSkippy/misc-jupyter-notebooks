import numpy as np
import pytest

from network_dynamics import theory


def test_giant_fraction_solves_equation():
    c = np.array([1.01, 1.5, 2, 3, 5, 10])
    s = theory.giant_fraction(c)
    assert s == pytest.approx(1 - np.exp(-c * s))
    assert (np.diff(s) > 0).all()


def test_no_giant_component_below_one():
    assert theory.giant_fraction([0, 0.5, 1]).tolist() == [0, 0, 0]


def test_known_values():
    assert float(theory.giant_fraction(2)) == pytest.approx(0.7968121300200202)
    assert float(theory.isolated_fraction(2)) == pytest.approx(np.exp(-2))
    assert float(theory.clustering(10, 1001)) == pytest.approx(0.01)
    assert theory.degree_pmf(np.arange(60), 4).sum() == pytest.approx(1)


def test_degree_ccdf():
    assert float(theory.degree_ccdf(0, 3)) == pytest.approx(1)
    assert float(theory.degree_ccdf(1, 3)) == pytest.approx(1 - np.exp(-3))


def test_preferential_ccdf():
    assert theory.preferential_ccdf([3, 5], 5).tolist() == [1, 1]
    assert float(theory.preferential_ccdf(10, 5)) == pytest.approx(30 / 110)
    # the degree probabilities implied by the tail sum to 1
    k = np.arange(5, 200_000)
    pmf = theory.preferential_ccdf(k, 5) - theory.preferential_ccdf(k + 1, 5)
    assert pmf.sum() == pytest.approx(1, abs=1e-6)


def test_preferential_degree():
    assert float(theory.preferential_degree(1, 100, 5)) == 50
    assert float(theory.preferential_degree(100, 100, 5)) == 5
