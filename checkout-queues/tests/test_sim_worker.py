import numpy as np
import pytest

from checkout_queues.sim_worker import random_inputs, simulate_line_lengths, simulate_store


@pytest.mark.parametrize("d", [1, 2, 4])
def test_array_version_matches_object_model(d):
    inputs = random_inputs(2000, 4, p_arrive=0.5, p_checkout=0.15, rng=1)
    df = simulate_store(d, *inputs)
    lengths = simulate_line_lengths(d, *inputs)
    np.testing.assert_allclose(df["avg_shoppers"], lengths.mean(axis=1))


def test_random_inputs_shapes():
    arrivals, orders, checkouts = random_inputs(10, 3, 0.5, 0.5, rng=0)
    assert arrivals.shape == (10,) and orders.shape == (10, 3) and checkouts.shape == (10, 3)
    assert (np.sort(orders, axis=1) == np.arange(3)).all()


def test_more_choices_shorter_lines():
    inputs = random_inputs(5000, 5, p_arrive=0.6, p_checkout=0.15, rng=2)
    assert simulate_line_lengths(5, *inputs).mean() < simulate_line_lengths(1, *inputs).mean()
