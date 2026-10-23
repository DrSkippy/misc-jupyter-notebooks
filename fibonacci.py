"""Fibonacci series utilities."""

from itertools import islice
from typing import Iterator, List


def fibonacci_gen() -> Iterator[int]:
    """Yield the Fibonacci series indefinitely: 0, 1, 1, 2, 3, 5, ..."""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b


def fibonacci(n: int) -> int:
    """Return the n-th Fibonacci number (0-indexed: F(0)=0, F(1)=1)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def fibonacci_series(count: int) -> List[int]:
    """Return the first `count` Fibonacci numbers."""
    if count < 0:
        raise ValueError("count must be non-negative")
    return list(islice(fibonacci_gen(), count))


if __name__ == "__main__":
    print(fibonacci_series(10))  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    print(fibonacci(50))         # 12586269025
