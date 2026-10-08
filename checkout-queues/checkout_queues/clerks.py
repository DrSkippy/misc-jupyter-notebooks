"""One shared line vs one line per clerk.

Customers arrive as a Poisson process and need a uniformly distributed service time. With `n_clerks` clerks:

- single queue: one FIFO line; the customer at its head goes to whichever clerk frees up first
- multiple queues: a line per clerk; an arriving customer joins the line with the fewest people (counting the one
  being served), ties broken at random, and never switches

Both layouts are first-come-first-served within a line, so each customer's service start is just
max(arrival, time their clerk frees up), computed in arrival order without an event queue.
"""

from collections import deque

import numpy as np
import pandas as pd


def generate_customers(num_customers, arrival_rate=0.5, min_service=1.0, max_service=5.0, rng=None):
    """Return (arrival_times, service_times) in minutes.

    Inter-arrival times are exponential with `arrival_rate` customers per minute; service times are uniform on
    [min_service, max_service].
    """
    rng = np.random.default_rng(rng)
    arrivals = np.cumsum(rng.exponential(1 / arrival_rate, num_customers))
    service = rng.uniform(min_service, max_service, num_customers)
    return arrivals, service


def single_queue(arrivals, service, n_clerks=3):
    """Service start times for one shared line feeding all clerks."""
    free_at = np.zeros(n_clerks)
    start = np.empty_like(arrivals)
    for i, (arrive, duration) in enumerate(zip(arrivals, service)):
        clerk = free_at.argmin()
        start[i] = max(arrive, free_at[clerk])
        free_at[clerk] = start[i] + duration
    return start


def multiple_queues(arrivals, service, n_clerks=3, rng=None):
    """Service start times when each clerk has a line and customers join the shortest."""
    rng = np.random.default_rng(rng)
    departures = [deque() for _ in range(n_clerks)]  # departure times of everyone in each line
    start = np.empty_like(arrivals)
    for i, (arrive, duration) in enumerate(zip(arrivals, service)):
        for line in departures:
            while line and line[0] <= arrive:
                line.popleft()
        lengths = np.array([len(line) for line in departures])
        clerk = rng.choice(np.flatnonzero(lengths == lengths.min()))
        line = departures[clerk]
        start[i] = max(arrive, line[-1]) if line else arrive
        line.append(start[i] + duration)
    return start


def wait_stats(arrivals, service, start):
    """Summary of one run: mean/median/std/max wait (time in line) and mean total time (wait + service)."""
    wait = start - arrivals
    return {
        "avg_wait": wait.mean(),
        "median_wait": np.median(wait),
        "std_wait": wait.std(ddof=1),
        "max_wait": wait.max(),
        "avg_total": (wait + service).mean(),
        "throughput": len(arrivals) / ((start + service).max() - arrivals.min()),
    }


def compare(num_customers=500, num_simulations=100, arrival_rate=1.2, n_clerks=3, seed=0):
    """Run both layouts on the same customers `num_simulations` times; one row per (simulation, layout)."""
    rng = np.random.default_rng(seed)
    rows = []
    for sim in range(num_simulations):
        arrivals, service = generate_customers(num_customers, arrival_rate, rng=rng)
        for layout, start in (("single queue", single_queue(arrivals, service, n_clerks)),
                              ("multiple queues", multiple_queues(arrivals, service, n_clerks, rng=rng))):
            rows.append({"simulation": sim, "layout": layout, "arrival_rate": arrival_rate,
                         **wait_stats(arrivals, service, start)})
    return pd.DataFrame(rows)
