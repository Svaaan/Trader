"""Shared fixtures.

The synthetic panel here is the reason most of these tests can run in under a
second and without a network. It is not a stand-in for real prices -- nothing is
concluded from it -- it exists so that properties like "a feature may not change
when more history arrives" can be checked exhaustively on data whose every value
is known.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# Long enough that the 252-day windows in features.py have room to warm up and
# still leave several hundred usable rows on both sides of a split.
DEFAULT_DAYS = 1400


def synthetic_prices(days=DEFAULT_DAYS, seed=7, start="2018-01-01"):
    """A price series with trend, noise and volume, deterministic per seed."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=days, name="date")

    steps = rng.normal(0.0004, 0.015, size=days)
    close = 100.0 * np.exp(np.cumsum(steps))
    spread = np.abs(rng.normal(0.008, 0.004, size=days)) * close

    return pd.DataFrame({
        "open": close - rng.normal(0, 0.003, days) * close,
        "high": close + spread,
        "low": close - spread,
        "close": close,
        "volume": rng.integers(1_000_000, 9_000_000, days).astype(float),
    }, index=dates)


def synthetic_panel(symbols=("AAA", "BBB", "CCC", "DDD", "EEE", "FFF"),
                    days=DEFAULT_DAYS, start="2018-01-01"):
    """Several symbols on one calendar, wide enough for cross-sectional ranks."""
    return {symbol: synthetic_prices(days=days, seed=index + 1, start=start)
            for index, symbol in enumerate(symbols)}


@pytest.fixture
def prices():
    return synthetic_prices()


@pytest.fixture
def panel():
    return synthetic_panel()


@pytest.fixture
def offline_spec():
    """A Spec with every block that needs the network switched off.

    Macro downloads index history and events downloads earnings calendars, so
    a test that wants determinism and no network asks for neither. The blocks
    have their own tests that stub what they need.
    """
    from trader import dataset

    return dataset.Spec(use_macro=False, use_events=False, use_cross=True)
