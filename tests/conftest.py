"""Shared pytest fixtures. Puts the repo root on sys.path so `import src...` works."""
import logging
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

logging.getLogger("src").setLevel(logging.WARNING)


def simulate_garch_prices(n=3000, omega=0.05, alpha=0.10, beta=0.85, seed=0, start=100.0):
    """Synthetic daily close prices whose percent log-returns follow a known GARCH(1,1)."""
    rng = np.random.default_rng(seed)
    h = omega / (1 - alpha - beta)
    r = np.empty(n)
    for t in range(n):
        r[t] = np.sqrt(h) * rng.standard_normal()
        h = omega + alpha * r[t] ** 2 + beta * h
    dates = pd.bdate_range("2015-01-01", periods=n + 1)
    prices = start * np.exp(np.concatenate([[0.0], np.cumsum(r / 100)]))
    return pd.Series(prices, index=dates)


@pytest.fixture
def garch_prices():
    return simulate_garch_prices()
