"""Shared fixtures for the test suite."""

import numpy as np
import pytest

from time_to_traffic_sim.gamma_params import fit_gamma_params


@pytest.fixture(scope="session")
def gamma_params() -> tuple[float, float]:
    """Solved gamma parameters for mode=7, median=45."""
    return fit_gamma_params(mode_days=7.0, median_days=45.0)


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded RNG for reproducible tests."""
    return np.random.default_rng(seed=0)
