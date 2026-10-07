"""Object model of a store: checkout lines, shoppers, and the store that holds the lines."""

import random

import numpy as np


class Line:
    """A FIFO checkout line."""

    def __init__(self, label):
        self.label = label
        self.shoppers = []

    def __len__(self):
        return len(self.shoppers)

    def join_line(self, t, shopper):
        shopper.join_time = t
        self.shoppers.append(shopper)
        return self

    def checkout(self, t):
        """Check out the shopper at the front of the line; return them, or None if the line is empty."""
        if not self.shoppers:
            return None
        shopper = self.shoppers.pop(0)
        shopper.checkout_time = t
        return shopper

    def __str__(self):
        return "\n  ".join([
            f"line: {self.label}",
            f"shoppers: {len(self.shoppers):2}{'*' * len(self.shoppers)}",
            "",
        ])


class Shopper:

    def __init__(self):
        self.join_time = 0
        self.checkout_time = 0

    def decision(self, t, select_from_lines):
        """Pick the shortest of the inspected lines (the first one on ties)."""
        return min(select_from_lines, key=len)

    def wait(self):
        return self.checkout_time - self.join_time


class Store:

    def __init__(self, n=5):
        self.lines = [Line(f"line_{i}") for i in range(n)]
        self.metrics = []
        self.customer_wait_metrics = []

    def random_line(self, k=1):
        """k distinct lines chosen at random."""
        return random.sample(self.lines, k)

    def index_line(self, indexes):
        return [self.lines[i] for i in indexes]

    def checkout(self, t, line_checkout_status):
        """Check out one shopper from every line whose flag is set.

        Returns one entry per line: the shopper checked out, or None.
        """
        return [line.checkout(t) if status else None
                for line, status in zip(self.lines, line_checkout_status)]

    def random_checkout(self, t, p):
        return self.checkout(t, np.random.random(len(self.lines)) < p)

    def report(self, t, cos, output=False):
        """Record [t, mean line length, mean wait of this step's checkouts, mean time-in-line per line].

        Means over nothing (no checkouts, an empty line) are recorded as NaN.
        """
        self.customer_wait_metrics = [c.wait() for c in cos if c is not None]
        avg_n = np.mean([len(line) for line in self.lines])
        avg_wait = np.mean(self.customer_wait_metrics) if self.customer_wait_metrics else np.nan
        avg_t = [np.mean([t - s.join_time for s in line.shoppers]) if line.shoppers else np.nan
                 for line in self.lines]
        self.metrics.append([t, avg_n, avg_wait] + avg_t)
        if output:
            for line in self.lines:
                print(line)
