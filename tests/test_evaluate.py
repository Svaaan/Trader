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


# --- the window that could actually be held --------------------------------

def test_the_executable_return_is_reported_separately():
    """Two windows, same rows, same positions -- only the holding differs."""
    days = 300
    positions = [[1, 1]] * days
    graded = [[0.004, 0.004]] * days        # the label's close-to-close move
    probs, labels, ret, dates, syms = panel_inputs(positions, graded)
    # All of the move happens overnight: nothing is left after the open.
    reachable = np.zeros_like(ret)

    result = evaluate.evaluate(probs, labels, ret, dates, syms,
                               executable_returns=reachable,
                               train_up_share=0.6, cost=0.0)

    assert result.strategy_daily > 0.003
    assert result.executable_daily == pytest.approx(0.0, abs=1e-9)
    assert result.execution_gap > 0


def overnight_edge(days=400, seed=0):
    """A book that profits close-to-close and earns nothing after the open.

    Long AAA, short BBB, every day. The graded return rewards exactly that
    with noise on top; the executable return is pure noise. This is the shape
    of the real finding -- gross edge entirely inside the overnight gap.
    """
    rng = np.random.default_rng(seed)
    positions = [[1, -1]] * days
    graded = [[0.004 + rng.normal(0, 0.006), -0.004 + rng.normal(0, 0.006)]
              for _ in range(days)]
    probs, labels, ret, dates, syms = panel_inputs(positions, graded)
    return probs, labels, ret, dates, syms, rng.normal(0, 0.01, len(ret))


def test_an_edge_that_lives_overnight_is_caught_by_the_gate():
    """The regression test for the finding this whole feature exists for.

    Close to close the strategy looks excellent. Held from the first open after
    the signal it earns nothing. The gate has to refuse it.
    """
    from trader import explain

    days = 400
    probs, labels, ret, dates, syms, reachable = overnight_edge(days)

    result = evaluate.evaluate(probs, labels, ret, dates, syms,
                               executable_returns=reachable,
                               train_up_share=0.5, cost=0.0)

    assert result.strategy_tstat > 3, "the graded version should look good"
    assert result.executable_tstat < 2, "the reachable version should not"
    assert result.execution_gap > 0.5

    trust = explain.assess(
        {**result.to_dict(), "effective_rows": 4000, "days": days,
         "accuracy": 0.56, "baseline_accuracy": 0.50, "up_rate": 0.5},
        controls={"noise_floor": {"spread": 0.004}},
        walk_forward={"folds_positive": 6, "folds_run": 6})

    assert not trust.trusted
    money = next(c for c in trust.checks if c["name"] == "makes_money")
    assert not money["passed"]
    assert "first open" in trust.reason


def test_omitting_the_executable_series_falls_back_rather_than_crashing():
    """Old callers still work; they just get the optimistic answer."""
    days = 200
    positions = [[1, 1]] * days
    returns = [[0.001, 0.001]] * days

    result = evaluate.evaluate(*panel_inputs(positions, returns),
                               train_up_share=0.6)

    assert result.executable_daily == result.strategy_daily
    assert result.execution_gap == 0.0


def test_the_verdict_names_the_overnight_gap_when_that_is_the_story():
    probs, labels, ret, dates, syms, reachable = overnight_edge()

    result = evaluate.evaluate(probs, labels, ret, dates, syms,
                               executable_returns=reachable,
                               train_up_share=0.5, cost=0.0)
    result.edge = 0.03

    assert "overnight gap" in evaluate.verdict(result)


# --- overlapping holding windows ---------------------------------------------
#
# Past one session, consecutive rows share most of the same move. A model whose
# positions persist -- and they do, because the features behind them change
# slowly -- is graded on nearly the same return again and again, and every
# statistic that counts those rows as independent reads high. Measured on noise
# before the fix: a t-statistic of 2 cleared in 39% of runs at h=5 and 53% at
# h=20, against the 5% it promises.

def sticky_noise(rng, horizon, days=500, names=10, keep=0.9):
    """The daily portfolio series of a model with no skill and sticky views."""
    daily = rng.normal(0.0, 0.01, size=(days + horizon, names))
    forward = np.stack([daily[t:t + horizon].sum(axis=0) for t in range(days)])

    view = np.empty((days, names))
    view[0] = rng.choice([-1.0, 1.0], names)
    for t in range(1, days):
        redraw = rng.random(names) > keep
        view[t] = np.where(redraw, rng.choice([-1.0, 1.0], names), view[t - 1])

    return pd.Series((view * forward).mean(axis=1))


def false_pass_rate(horizon, *, told, trials=300, seed=0):
    rng = np.random.default_rng(seed)
    passes = 0
    for _ in range(trials):
        series = sticky_noise(rng, horizon)
        passes += abs(evaluate._tstat(series, told)) > 2.0
    return passes / trials


def test_the_overlap_problem_is_real_before_the_correction():
    """Without this the next test could pass on data that never overlapped."""
    assert false_pass_rate(5, told=1) > 0.25


def test_a_skill_free_model_no_longer_clears_two_at_long_horizons():
    assert false_pass_rate(5, told=5) < 0.10
    assert false_pass_rate(20, told=20) < 0.10


def test_nothing_changes_at_one_session():
    """Every number a one-day run has ever printed stays what it was."""
    rng = np.random.default_rng(1)
    series = sticky_noise(rng, 1)
    assert evaluate._overlap(series, 1) == 1.0
    assert evaluate._tstat(series, 1) == pytest.approx(
        series.mean() / (series.std() / np.sqrt(len(series))))


def test_the_correction_never_claims_overlap_added_information():
    rng = np.random.default_rng(2)
    for _ in range(50):
        assert evaluate._overlap(pd.Series(rng.normal(size=40)), 10) >= 1.0


def test_a_year_of_five_day_holds_is_fifty_periods_not_two_hundred():
    rng = np.random.default_rng(3)
    series = pd.Series(rng.normal(0.002, 0.02, 400))
    assert evaluate._sharpe(series, 5) == pytest.approx(
        evaluate._sharpe(series, 1) / np.sqrt(5))


def test_the_horizon_reaches_every_statistic_that_needs_it():
    """Through evaluate, not just the helpers: a Sharpe that dropped by the
    square root of five while the t-statistic and effective rows stayed put
    would mean one call site had been missed."""
    rng = np.random.default_rng(4)
    days, names, horizon = 400, 8, 5
    daily = rng.normal(0.0, 0.01, size=(days + horizon, names))
    forward = np.stack([daily[t:t + horizon].sum(axis=0) for t in range(days)])
    view = np.empty((days, names))
    view[0] = rng.random(names)
    for t in range(1, days):
        view[t] = np.where(rng.random(names) > 0.9, rng.random(names), view[t - 1])

    calendar = pd.bdate_range("2020-01-01", periods=days)
    args = (view.ravel(), (forward.ravel() > 0).astype(int), forward.ravel(),
            np.repeat(calendar, names),
            np.tile([f"S{i}" for i in range(names)], days))

    naive = evaluate.evaluate(*args, train_up_share=0.5, cost=0.0)
    told = evaluate.evaluate(*args, train_up_share=0.5, cost=0.0, horizon=horizon)

    assert told.executable_sharpe == pytest.approx(
        naive.executable_sharpe / np.sqrt(horizon), abs=2e-3)
    assert abs(told.executable_tstat) <= abs(naive.executable_tstat)
    assert told.effective_rows < naive.effective_rows
    assert told.executable_daily == pytest.approx(
        naive.executable_daily / horizon, abs=1e-6)
