"""Grading, and the three ways it used to flatter the model.

Each of these is a regression test for a measured mistake:

**Cost was charged per row rather than per trade.** The model held long on 99%
of days and changed position 113 times in 4,187 rows, and was charged 4,187
round trips. That is 14.9 points of annual return against a true 0.85, and it
turned a strategy that roughly matched buy-and-hold into one that appeared to
destroy a sixth of the account every year.

**Confidence buckets printed accuracy on five rows.** One stored run reads
"accuracy 1.0" on a bucket of five. In a project whose entire purpose is
refusing to overclaim, that was the single most misleading line on the page.

**Returns carried no uncertainty at all.** The gate computes a standard error
for accuracy and refuses to speak without one; the return figures next to it
were printed bare.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trader import evaluate                                   # noqa: E402


def panel_inputs(positions_per_day, returns_per_day, symbols=("AAA", "BBB")):
    """Build the (probabilities, labels, returns, dates, symbols) tuple.

    `positions_per_day` is a list of lists of +1/-1 -- what the model says for
    each symbol on each day.
    """
    dates, syms, probs, rets = [], [], [], []
    calendar = pd.bdate_range("2024-01-01", periods=len(positions_per_day))

    for day, (row, returns) in enumerate(zip(positions_per_day, returns_per_day)):
        for index, symbol in enumerate(symbols):
            dates.append(calendar[day])
            syms.append(symbol)
            probs.append(0.9 if row[index] > 0 else 0.1)
            rets.append(returns[index])

    probs = np.array(probs)
    rets = np.array(rets)
    labels = (rets > 0).astype(int)
    return probs, labels, rets, pd.DatetimeIndex(dates), np.array(syms)


# --- cost -------------------------------------------------------------------

def test_a_position_held_unchanged_pays_no_further_cost():
    """The regression test for a 17x overcharge."""
    days = 200
    positions = [[1, 1]] * days                    # long every day, never traded
    returns = [[0.001, 0.001]] * days

    result = evaluate.evaluate(*panel_inputs(positions, returns),
                               train_up_share=0.6, cost=0.0005)

    # One entry, then nothing. Turnover averaged over 200 days is ~1/200.
    assert result.turnover_daily < 0.02, (
        "a position that never changes is being charged as though it did")
    # And the cost drag must be a fraction of a point, not fifteen.
    assert result.cost_drag_annualised < 0.01


def test_a_position_that_flips_every_day_pays_every_day():
    days = 200
    positions = [[1, 1] if d % 2 == 0 else [-1, -1] for d in range(days)]
    returns = [[0.001, 0.001]] * days

    result = evaluate.evaluate(*panel_inputs(positions, returns),
                               train_up_share=0.6, cost=0.0005)

    # Flipping the whole book daily is a full round trip: two units of weight.
    assert result.turnover_daily > 1.5
    assert result.cost_drag_annualised > 0.15


def test_cost_scales_with_the_rate_charged():
    days = 100
    positions = [[1, -1] if d % 3 else [-1, 1] for d in range(days)]
    returns = [[0.002, -0.001]] * days

    cheap = evaluate.evaluate(*panel_inputs(positions, returns),
                              train_up_share=0.5, cost=0.0001)
    dear = evaluate.evaluate(*panel_inputs(positions, returns),
                             train_up_share=0.5, cost=0.005)
    assert dear.strategy_daily < cheap.strategy_daily


# --- buckets ----------------------------------------------------------------

def test_a_bucket_with_too_few_rows_withholds_its_accuracy():
    """"100% accurate on its five most confident days" is not a result."""
    rng = np.random.default_rng(0)
    rows = 400
    dates = pd.bdate_range("2024-01-01", periods=rows // 2).repeat(2)
    symbols = np.array(["AAA", "BBB"] * (rows // 2))

    # Almost everything at a coin flip; a handful of very confident calls.
    probs = np.full(rows, 0.51)
    probs[:5] = 0.99
    returns = rng.normal(0, 0.01, rows)
    labels = (returns > 0).astype(int)

    result = evaluate.evaluate(probs, labels, returns, dates, symbols,
                               train_up_share=0.5)

    confident = result.by_confidence[-1]
    assert confident["rows"] == 5
    assert confident["enough"] is False
    assert confident["accuracy"] is None, (
        "an accuracy on five rows was printed as though it meant something")
    # But the interval is still offered, and it has to be honestly wide.
    low, high = confident["accuracy_interval"]
    assert high - low > 0.4


def test_a_bucket_with_enough_rows_reports_its_accuracy():
    rows = 600
    dates = pd.bdate_range("2024-01-01", periods=rows // 2).repeat(2)
    symbols = np.array(["AAA", "BBB"] * (rows // 2))
    probs = np.full(rows, 0.95)
    returns = np.full(rows, 0.01)
    labels = np.ones(rows, dtype=int)

    result = evaluate.evaluate(probs, labels, returns, dates, symbols,
                               train_up_share=0.5)
    confident = result.by_confidence[-1]
    assert confident["rows"] == rows
    assert confident["accuracy"] == 1.0


def test_bucket_returns_are_net_like_the_headline():
    """Gross in one place and net in the other invites comparing them."""
    days = 100
    # Flipping every day, so every row genuinely pays to trade.
    positions = [[1, 1] if d % 2 else [-1, -1] for d in range(days)]
    returns = [[0.0, 0.0]] * days          # zero gross, so only cost remains

    result = evaluate.evaluate(*panel_inputs(positions, returns),
                               train_up_share=0.6, cost=0.001)
    populated = [b for b in result.by_confidence if b["rows"]]
    assert populated
    for bucket in populated:
        assert bucket["mean_net_return"] < 0, (
            "a bucket earning nothing gross should still show the cost")


def test_bucket_costs_are_charged_on_the_trade_not_on_the_row():
    """The same overcharge, in the place it was easiest to leave behind.

    A book held unchanged pays once at entry. Charging every row would show a
    steady loss on a position that never traded -- which is exactly what the
    headline used to do.
    """
    days = 200
    positions = [[1, 1]] * days
    returns = [[0.0, 0.0]] * days

    result = evaluate.evaluate(*panel_inputs(positions, returns),
                               train_up_share=0.6, cost=0.001)
    populated = [b for b in result.by_confidence if b["rows"]]
    assert populated
    for bucket in populated:
        # One entry amortised over 200 days, not 200 round trips.
        assert bucket["mean_net_return"] > -0.001 / 10, (
            "a carried position was charged as though it traded every day")


# --- the baseline -----------------------------------------------------------

def test_the_baseline_comes_from_training_not_from_the_test_period():
    """A baseline read off the test set is one nobody had in advance."""
    rows = 400
    dates = pd.bdate_range("2024-01-01", periods=rows // 2).repeat(2)
    symbols = np.array(["AAA", "BBB"] * (rows // 2))
    probs = np.full(rows, 0.6)
    # The test period is 70% down, but training was mostly up.
    labels = np.zeros(rows, dtype=int)
    labels[: int(rows * 0.3)] = 1
    returns = np.where(labels == 1, 0.01, -0.01)

    result = evaluate.evaluate(probs, labels, returns, dates, symbols,
                               train_up_share=0.8)

    # Always-up, the training majority, scores 30% here.
    assert result.baseline_accuracy == pytest.approx(0.3, abs=0.01)
    # The test period's own majority is 70% and is reported separately.
    assert result.test_majority == pytest.approx(0.7, abs=0.01)
    assert "training majority" in result.baseline_source


# --- uncertainty ------------------------------------------------------------

def test_returns_carry_a_t_statistic_and_a_drawdown():
    rng = np.random.default_rng(3)
    days = 300
    positions = [[1, 1]] * days
    returns = [list(rng.normal(0.0002, 0.01, 2)) for _ in range(days)]

    result = evaluate.evaluate(*panel_inputs(positions, returns),
                               train_up_share=0.6)

    assert isinstance(result.strategy_tstat, float)
    assert isinstance(result.strategy_sharpe, float)
    assert result.strategy_max_drawdown <= 0.0


def test_a_strong_but_short_record_is_not_called_an_edge():
    """The verdict has to refuse on the t-statistic, not just on accuracy."""
    rng = np.random.default_rng(1)
    days = 200
    # Varying, so it does not trip the stuck-model guard before reaching the
    # question this test is actually about.
    positions = [[1, 1] if d % 2 else [-1, -1] for d in range(days)]
    returns = [list(rng.normal(0.0001, 0.02, 2)) for _ in range(days)]

    result = evaluate.evaluate(*panel_inputs(positions, returns),
                               train_up_share=0.5)
    result.edge = 0.05          # pretend a large accuracy edge
    result.strategy_tstat = 0.4
    assert "t-statistic" in evaluate.verdict(result)


def test_effective_rows_discount_a_panel_that_moves_together():
    """Ten symbols agreeing every day are not ten independent verdicts."""
    days = 200
    # Both symbols always right or always wrong together: perfectly correlated.
    positions = [[1, 1]] * days
    returns = [[0.01, 0.01] if d % 2 else [-0.01, -0.01] for d in range(days)]

    result = evaluate.evaluate(*panel_inputs(positions, returns),
                               train_up_share=0.6)

    assert result.rows == days * 2
    assert result.design_effect > 1.5
    assert result.effective_rows < result.rows, (
        "a perfectly correlated panel was counted as independent")


def test_independent_symbols_are_barely_discounted():
    rng = np.random.default_rng(5)
    days = 400
    positions = [[1, 1]] * days
    returns = [list(rng.normal(0, 0.01, 2)) for _ in range(days)]

    result = evaluate.evaluate(*panel_inputs(positions, returns),
                               train_up_share=0.6)
    assert result.design_effect < 1.4


def test_annualising_compounds_the_real_series_not_its_mean():
    """A mean raised to the 252nd power ignores volatility drag."""
    rng = np.random.default_rng(7)
    days = 504                                    # two years
    positions = [[1, 1]] * days
    daily = rng.normal(0.0005, 0.03, days)        # deliberately volatile
    returns = [[d, d] for d in daily]

    result = evaluate.evaluate(*panel_inputs(positions, returns),
                               train_up_share=0.6, cost=0.0)

    naive = (1.0 + float(np.mean(daily))) ** 252 - 1.0
    assert result.strategy_annualised < naive, (
        "compounding the mean overstates a volatile series")


def test_mismatched_inputs_are_refused():
    with pytest.raises(ValueError):
        evaluate.evaluate([0.5, 0.5], [1], [0.01, 0.01],
                          pd.bdate_range("2024-01-01", periods=2), ["A", "B"])
