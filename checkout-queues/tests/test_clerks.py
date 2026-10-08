import numpy as np

from checkout_queues import clerks


def test_no_wait_when_customers_are_spread_out():
    arrivals = np.array([0.0, 10.0, 20.0])
    service = np.array([5.0, 5.0, 5.0])
    for start in (clerks.single_queue(arrivals, service), clerks.multiple_queues(arrivals, service, rng=0)):
        np.testing.assert_array_equal(start, arrivals)


def test_single_queue_with_one_clerk_is_fifo():
    arrivals = np.array([0.0, 1.0, 2.0])
    service = np.array([3.0, 3.0, 3.0])
    np.testing.assert_array_equal(clerks.single_queue(arrivals, service, n_clerks=1), [0.0, 3.0, 6.0])


def test_layouts_agree_with_one_clerk():
    arrivals, service = clerks.generate_customers(200, arrival_rate=0.3, rng=0)
    np.testing.assert_allclose(clerks.single_queue(arrivals, service, 1),
                               clerks.multiple_queues(arrivals, service, 1, rng=0))


def test_single_queue_waits_less_on_average():
    df = clerks.compare(num_customers=300, num_simulations=20, arrival_rate=1.2)
    means = df.groupby("layout")["avg_wait"].mean()
    assert means["single queue"] < means["multiple queues"]
