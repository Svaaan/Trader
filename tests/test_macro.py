"""Macro series that stop, and the difference between a holiday and an ending.

The macro block puts every series on one calendar and forward-fills, on the
grounds that a holiday means the level did not change. That is true for a day
or two. It was not true for ^VIX3M, which the provider stopped updating on
2026-07-17: every session since carried the July close, and `vix_term_slope`
divided a two-month-old number by a current one -- 39 rows of it, on the end of
the sealed test period and on every live signal.

The obvious repair has a trap in it. `build` drops any date with a missing
column, and the panel drops any row whose macro join is empty, so simply
leaving the stopped series blank would delete the most recent sessions for
every symbol. These check both halves: nothing is carried past a holiday's
length, and the recent rows survive.

`prices.load` is stubbed throughout, so none of this touches the network.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from conftest import synthetic_prices                         # noqa: E402
from trader import macro                                      # noqa: E402

DAYS = 600

# Deliberately not macro.MAX_FILL_SESSIONS: these have to be able to fail
# against a version of the module that has no such constant.
WELL_PAST_A_HOLIDAY = 10


@pytest.fixture(autouse=True)
def fresh_warnings(monkeypatch):
    # Warnings are said once per process; each test is its own process here.
    monkeypatch.setattr(macro, "_WARNED", set(), raising=False)


def provider(monkeypatch, *, ends_early=None, holes=None):
    """Every macro series on one calendar, with some of them cut or holed.

    `ends_early` maps a series name to how many sessions before the others it
    stops; `holes` maps a name to (start position, length) of an interior gap.
    """
    series = {}
    for seed, (name, ticker) in enumerate(macro.SERIES.items(), start=1):
        frame = synthetic_prices(days=DAYS, seed=seed)
        if ends_early and name in ends_early:
            frame = frame.iloc[:-ends_early[name]]
        if holes and name in holes:
            start, length = holes[name]
            frame = frame.drop(frame.index[start:start + length])
        series[ticker] = frame

    def load(ticker, *, period="10y", refresh=False):
        return series[ticker]

    monkeypatch.setattr(macro.prices_mod, "load", load)
    calendar = synthetic_prices(days=DAYS, seed=1).index
    return calendar, series


def values_after(block, column, date, sessions):
    """Non-empty values of `column` more than `sessions` after `date`."""
    if column not in block.columns:
        return pd.Series(dtype="float64")
    later = block.index[block.index > date]
    return block.loc[later[sessions:], column].dropna()


# --- the ending ---------------------------------------------------------------

def test_a_series_that_stops_is_not_carried_forward(monkeypatch):
    calendar, series = provider(monkeypatch, ends_early={"vix3m": 40})
    stopped = series["^VIX3M"].index[-1]

    block = macro.build()

    carried = values_after(block, "vix_term_slope", stopped,
                           WELL_PAST_A_HOLIDAY + macro.LAG_SESSIONS)
    assert carried.empty, (
        f"vix_term_slope has {len(carried)} values built from a VIX3M close "
        f"that stopped on {stopped.date()}")


def test_a_stopped_series_does_not_freeze_its_returns_either(monkeypatch):
    """pct_change pads by default, so a return over a frozen series reads as
    exactly zero rather than as missing -- the same fill, one step later."""
    calendar, series = provider(monkeypatch, ends_early={"oil": 40})
    stopped = series["CL=F"].index[-1]

    block = macro.build()

    carried = values_after(block, "oil_return_20d", stopped,
                           WELL_PAST_A_HOLIDAY + macro.LAG_SESSIONS)
    assert carried.empty, (
        f"oil_return_20d has {len(carried)} values after oil stopped, "
        f"{int((carried == 0.0).sum())} of them exactly zero")


def test_the_recent_rows_survive_a_series_ending(monkeypatch):
    """The trap: a blank column in the block deletes that date for every
    symbol in the panel, and the most recent date is the live signal."""
    calendar, _ = provider(monkeypatch, ends_early={"vix3m": 40})

    block = macro.build()

    assert block.index[-1] == calendar[-1]
    # Every one of the last sixty sessions, including the forty after it stopped.
    assert len(block.loc[calendar[-60]:]) == 60


def test_an_ended_series_is_named_with_the_date_it_stopped(monkeypatch):
    _, series = provider(monkeypatch, ends_early={"vix3m": 40})

    report = {}
    block = macro.build(report=report)

    assert "vix_term_slope" not in block.columns
    assert "vix3m" in report["ended"]
    ended = report["ended"]["vix3m"]
    assert ended["ticker"] == "^VIX3M"
    assert ended["last"] == str(series["^VIX3M"].index[-1].date())
    assert ended["sessions_behind"] == 40


def test_the_ending_is_judged_against_the_other_series_not_the_clock(
        monkeypatch):
    """A whole block that is a weekend behind has not ended; one series forty
    sessions behind the rest has."""
    provider(monkeypatch)
    report = {}
    block = macro.build(report=report)
    assert report["ended"] == {}
    assert set(block.columns) == set(macro.MACRO_NAMES)


# --- a holiday, and an outage ------------------------------------------------------

def test_a_holiday_length_gap_is_still_filled(monkeypatch):
    """What the forward-fill was for: STOXX shut for Christmas while New York
    traded. The level did not change, and the rows must not go."""
    calendar, _ = provider(monkeypatch, holes={"stoxx": (400, 2)})

    report = {}
    block = macro.build(report=report)

    closed = calendar[400:402] + pd.offsets.BDay(macro.LAG_SESSIONS)
    assert all(date in block.index for date in closed)
    assert block.loc[closed, "stoxx_return_5d"].notna().all()
    assert report["gaps"] == {}


def test_an_outage_longer_than_a_holiday_is_left_blank_and_counted(
        monkeypatch, caplog):
    """The provider lost eight sessions of the dollar. The first few are filled
    as a holiday would be; the rest are not known, are not invented, and are
    reported -- because a blank there costs those dates for every symbol."""
    calendar, _ = provider(monkeypatch, holes={"dollar": (400, 8)})

    report = {}
    with caplog.at_level("WARNING", logger="trader.macro"):
        block = macro.build(report=report)

    blank = 8 - macro.MAX_FILL_SESSIONS
    assert report["gaps"] == {"dollar": blank}
    unknown = calendar[400 + macro.MAX_FILL_SESSIONS:408]
    assert not any((date + pd.offsets.BDay(macro.LAG_SESSIONS)) in block.index
                   for date in unknown)
    assert any("dollar" in r.getMessage() and "blank" in r.getMessage()
               for r in caplog.records)
    # And the series is not treated as ended: it resumed.
    assert "dollar_return_20d" in block.columns
    assert report["ended"] == {}


def test_the_fill_limit_covers_every_gap_the_real_series_have():
    """Measured on the cached ten-year histories: the longest interior gap in
    any macro series was two sessions (STOXX, Christmas 2018). A limit below
    that would start deleting history."""
    assert macro.MAX_FILL_SESSIONS >= 2
    assert macro.MAX_FILL_SESSIONS < WELL_PAST_A_HOLIDAY
