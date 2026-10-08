"""Power-of-d-choices store simulations.

Every step at most one shopper arrives, inspects d lines and joins the shortest; then each non-empty line checks
out its front shopper with probability p_checkout.

The random inputs are drawn once by `random_inputs` and shared by every value of d (common random numbers), so
differences between values of d come from the policy, not from different luck.
"""

import numpy as np
import pandas as pd

from checkout_queues.stores import Shopper, Store


def random_inputs(n_steps, n_lines, p_arrive, p_checkout, rng=None):
    """Draw the shared random inputs for a run.

    Returns (arrivals, line_orders, checkouts):
      arrivals     (n_steps,) bool: a shopper arrives at step t
      line_orders  (n_steps, n_lines) int: a random order of the lines; a shopper inspecting d lines looks at the
                   first d
      checkouts    (n_steps, n_lines) bool: line i checks out a shopper at step t (if anyone is in it)
    """
    rng = np.random.default_rng(rng)
    arrivals = rng.random(n_steps) < p_arrive
    line_orders = rng.permuted(np.tile(np.arange(n_lines), (n_steps, 1)), axis=1)
    checkouts = rng.random((n_steps, n_lines)) < p_checkout
    return arrivals, line_orders, checkouts


def simulate_store(d, arrivals, line_orders, checkouts):
    """Run the object model for shoppers who inspect d lines.

    Returns a DataFrame with one row per step: time, avg_shoppers (mean line length), avg_wait (mean wait of the
    shoppers checked out that step) and line_<i> (mean time-in-line of the shoppers in line i).
    """
    n_lines = line_orders.shape[1]
    store = Store(n_lines)
    for t, (arrive, order, checkout) in enumerate(zip(arrivals, line_orders, checkouts)):
        if arrive:
            shopper = Shopper()
            shopper.decision(t, store.index_line(order[:d])).join_line(t, shopper)
        store.report(t, store.checkout(t, checkout))
    return pd.DataFrame(store.metrics,
                        columns=["time", "avg_shoppers", "avg_wait"] + [f"line_{i}" for i in range(n_lines)])


def simulate_line_lengths(d, arrivals, line_orders, checkouts):
    """Array-only version of `simulate_store` that tracks just the line lengths.

    Returns an (n_steps, n_lines) int array: the length of every line at the end of every step. Given the same
    inputs it matches `simulate_store` exactly.
    """
    n_steps, n_lines = line_orders.shape
    lengths = np.zeros((n_steps, n_lines), dtype=int)
    current = np.zeros(n_lines, dtype=int)
    for t in range(n_steps):
        if arrivals[t]:
            inspected = line_orders[t, :d]
            current[inspected[np.argmin(current[inspected])]] += 1
        current -= checkouts[t] & (current > 0)
        lengths[t] = current
    return lengths
