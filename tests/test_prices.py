"""The price cache, and the difference between a stale cache and a stale provider.

The cache refetches when it covers less than was asked for or ends more than a
few days ago. Both rules assume that asking again can help. For ^VIX3M, which
the provider stopped updating on 2026-07-17, and for DSFIR.AS, which listed in
2023, it cannot -- and every load refetched them, got back exactly what was
cached, and refetched again at the next assembly: nine times in one search.

The network is stubbed throughout. `load` imports yfinance inside the function,
so a fake module in sys.modules is what it gets, and every call to it is counted.
"""

import datetime as dt
import json
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trader import prices                                     # noqa: E402


TODAY = dt.datetime.now(dt.timezone.utc).date()


def history(start, end):
    """What yfinance returns: capitalised columns on a tz-aware index."""
    index = pd.bdate_range(start, end, tz="America/New_York")
    close = 20.0 + np.cumsum(np.random.default_rng(0).normal(0, 0.2, len(index)))
    return pd.DataFrame({"Open": close, "High": close + 0.1, "Low": close - 0.1,
                         "Close": close, "Volume": 0.0}, index=index)


class Provider:
    """A stand-in for yfinance that only ever has one fixed history."""

    def __init__(self, frame):
        self.frame = frame
        self.calls = 0

    def module(self):
        provider = self

        class Ticker:
            def __init__(self, symbol):
                self.symbol = symbol

            def history(self, **_):
                provider.calls += 1
                return provider.frame.copy()

        return types.SimpleNamespace(Ticker=Ticker)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(prices, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(prices, "_WARNED", set(), raising=False)
    return tmp_path


def provide(monkeypatch, frame):
    provider = Provider(frame)
    monkeypatch.setitem(sys.modules, "yfinance", provider.module())
    return provider


# --- the loop this fixes --------------------------------------------------------

def test_a_series_the_provider_stopped_updating_is_fetched_once_not_every_load(
        cache, monkeypatch):
    """^VIX3M: ten years that end two months ago, at the source."""
    frozen = TODAY - dt.timedelta(days=58)
    provider = provide(monkeypatch,
                       history(frozen - dt.timedelta(days=3650), frozen))

    first = prices.load("^VIX3M", period="10y")
    assert provider.calls == 1

    for _ in range(3):
        again = prices.load("^VIX3M", period="10y")
    assert provider.calls == 1, "asked the provider again for rows it does not have"
    pd.testing.assert_frame_equal(first, again, check_freq=False)


def test_a_listing_younger_than_the_period_is_fetched_once(cache, monkeypatch):
    """DSFIR.AS: current, but three years old, and asked for ten."""
    provider = provide(monkeypatch, history("2023-04-18",
                                            TODAY - dt.timedelta(days=2)))

    prices.load("DSFIR.AS", period="10y")
    prices.load("DSFIR.AS", period="10y")
    assert provider.calls == 1


def test_a_short_but_current_listing_is_not_warned_about_forward_filling(
        cache, monkeypatch, caplog):
    provide(monkeypatch, history("2023-04-18", TODAY - dt.timedelta(days=2)))
    prices.load("DSFIR.AS", period="10y")
    with caplog.at_level("WARNING", logger="trader.prices"):
        prices.load("DSFIR.AS", period="10y")

    said = [r.getMessage() for r in caplog.records
            if "provider only has" in r.getMessage()]
    assert len(said) == 1
    assert "short of 10y" in said[0]
    assert "forward-fills" not in said[0] and "out of date" not in said[0]


def test_the_provider_limit_is_said_once_per_process(cache, monkeypatch, caplog):
    frozen = TODAY - dt.timedelta(days=58)
    provide(monkeypatch, history(frozen - dt.timedelta(days=3650), frozen))

    last = prices.load("^VIX3M", period="10y").index[-1].date()
    with caplog.at_level("WARNING", logger="trader.prices"):
        prices.load("^VIX3M", period="10y")
        prices.load("^VIX3M", period="10y")

    said = [r.getMessage() for r in caplog.records
            if "provider only has" in r.getMessage()]
    assert len(said) == 1
    assert str(last) in said[0] and "forward-fills" in said[0]


# --- and the limits on believing it ---------------------------------------------

def test_it_asks_again_once_the_recheck_interval_has_passed(cache, monkeypatch):
    """A series that resumes updating has to be picked up."""
    frozen = TODAY - dt.timedelta(days=58)
    provider = provide(monkeypatch,
                       history(frozen - dt.timedelta(days=3650), frozen))
    prices.load("^VIX3M", period="10y")

    path = os.path.join(str(cache), prices.FETCH_LOG)
    log = json.load(open(path, encoding="utf-8"))
    earlier = (dt.datetime.now(dt.timezone.utc)
               - dt.timedelta(hours=prices.RECHECK_HOURS + 1))
    log["^VIX3M"]["at"] = earlier.isoformat(timespec="seconds")
    json.dump(log, open(path, "w", encoding="utf-8"))

    prices.load("^VIX3M", period="10y")
    assert provider.calls == 2


def test_a_shorter_fetch_cannot_vouch_for_a_longer_request(cache, monkeypatch):
    """The original AAPL bug: two years cached, ten asked for, never noticed."""
    provider = provide(monkeypatch, history(TODAY - dt.timedelta(days=730),
                                            TODAY - dt.timedelta(days=2)))
    prices.load("AAPL", period="2y")
    assert provider.calls == 1

    prices.load("AAPL", period="10y")
    assert provider.calls == 2


def test_a_cache_replaced_since_the_fetch_is_not_taken_on_the_log_word(
        cache, monkeypatch):
    frozen = TODAY - dt.timedelta(days=58)
    provider = provide(monkeypatch,
                       history(frozen - dt.timedelta(days=3650), frozen))
    prices.load("^VIX3M", period="10y")

    csv = os.path.join(str(cache), "^VIX3M.csv")
    trimmed = pd.read_csv(csv, index_col="date").iloc[:-5]
    trimmed.to_csv(csv)

    prices.load("^VIX3M", period="10y")
    assert provider.calls == 2


def test_a_healthy_cache_is_served_without_asking(cache, monkeypatch):
    provider = provide(monkeypatch, history(TODAY - dt.timedelta(days=3652),
                                            TODAY - dt.timedelta(days=1)))
    prices.load("MSFT", period="10y")
    prices.load("MSFT", period="10y")
    assert provider.calls == 1


def test_a_log_that_cannot_be_written_does_not_stop_prices_loading(
        cache, monkeypatch):
    provider = provide(monkeypatch, history(TODAY - dt.timedelta(days=3652),
                                            TODAY - dt.timedelta(days=1)))

    def refuse(*_args, **_kwargs):
        raise PermissionError("the log is open somewhere else")

    monkeypatch.setattr(prices.os, "replace", refuse)
    frame = prices.load("MSFT", period="10y")
    assert len(frame) > 2000
    assert provider.calls == 1
