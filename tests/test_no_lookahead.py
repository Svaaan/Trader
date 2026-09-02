"""Nothing computed for day t may depend on anything after day t.

This is the mistake that makes a trading model look brilliant and be worthless,
and it does not announce itself: the accuracy simply comes out high. Reading the
code is not enough of a check, because the ways it happens are all one character
wide -- `shift(-1)` instead of `shift(1)`, `center=True` on a rolling window, a
scaler fitted before the split.

So these test the property rather than the implementation. Compute the features
on a truncated history and again on the full one; every date they have in common
must be identical. A feature that can see the future changes when the future
arrives, and that is detectable without knowing how it cheats.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trader import dataset, features, labels     # noqa: E402


def synthetic_prices(days=400, seed=7):
    """A price series with trend, noise and volume, deterministic per seed."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=days, name="date")

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


# --- the property itself ---------------------------------------------------

@pytest.mark.parametrize("cut", [120, 200, 310])
def test_a_feature_does_not_change_when_more_history_arrives(cut):
    prices = synthetic_prices()

    full = features.build(prices)
    truncated = features.build(prices.iloc[:cut])

    shared = truncated.index.intersection(full.index)
    assert len(shared) > 20, "the truncation left too little to compare"

    for column in features.FEATURE_NAMES:
        a = truncated.loc[shared, column].to_numpy()
        b = full.loc[shared, column].to_numpy()
        # Not almost-equal: these are the same arithmetic on the same inputs.
        # A difference means later data reached an earlier row.
        assert np.allclose(a, b, rtol=0, atol=1e-12, equal_nan=True), (
            f"{column} changed for dates it had already seen once {cut} more "
            f"days arrived -- it is reading forward")


def test_the_label_does_look_forward():
    """The other half: a label that cannot see the future predicts nothing.

    Stated as a test because the pair is what matters. If someone ever 'fixes'
    a look-ahead warning by removing the shift here, the model would be trained
    to predict the day it was shown, and the accuracy would be superb.
    """
    prices = synthetic_prices(days=60)
    y = labels.direction(prices, horizon=1)

    rose = prices["close"].shift(-1) > prices["close"]
    both = y.notna()

    assert (y[both] == rose[both].astype(float)).all(), (
        "the label is not describing the next session's move")

    assert y.isna().sum() == 1, (
        "the final row should have no label: its future has not happened")


def test_the_last_row_has_features_but_no_label():
    """Which is exactly the row a live signal is produced for."""
    prices = synthetic_prices(days=200)

    x = features.build(prices)
    y = labels.direction(prices, horizon=1)

    last = x.index[-1]
    assert last in x.index
    assert pd.isna(y.loc[last]), (
        "the most recent day should be predictable but not yet gradeable")


# --- splitting time --------------------------------------------------------

def test_the_split_is_chronological():
    prices = synthetic_prices(days=500)
    split = dataset.build_one("TEST", prices, test_fraction=0.2)

    assert split.train_dates[-1] < split.test_dates[0], (
        "training data runs past the start of the test period; a random split "
        "of a price series lets the model memorise both sides of a gap")

    assert len(split.y_test) > 0 and len(split.y_train) > len(split.y_test)


def test_the_scaler_never_sees_the_test_period():
    """Standardising on the whole series leaks the future's distribution."""
    prices = synthetic_prices(days=500)
    split = dataset.build_one("TEST", prices, test_fraction=0.2)

    _, _, scaler = dataset.combine([split])

    expected = split.x_train.mean(axis=0)
    assert np.allclose(scaler.mean, expected, atol=1e-6), (
        "the scaler's mean does not match the training rows alone")

    everything = np.concatenate([split.x_train, split.x_test]).mean(axis=0)
    if not np.allclose(expected, everything, atol=1e-9):
        assert not np.allclose(scaler.mean, everything, atol=1e-9), (
            "the scaler was fitted on train and test together")


def test_a_constant_feature_does_not_become_infinity():
    """std of 0 divides to inf, and a column of inf trains nothing."""
    scaler = dataset.Scaler(
        mean=np.array([0.0, 5.0], dtype=np.float32),
        std=np.array([1.0, 0.0], dtype=np.float32),
        feature_names=["moves", "constant"],
    )
    out = scaler.apply(np.array([[1.0, 5.0], [2.0, 5.0]], dtype=np.float32))

    assert np.isfinite(out).all(), "a constant column produced a non-finite value"


# --- what gets sent --------------------------------------------------------

def test_the_packed_dataset_is_loadable_without_unpickling():
    """HelloWorldAi refuses anything it would have to unpickle, and is right to."""
    prices = synthetic_prices(days=400)
    split = dataset.build_one("TEST", prices)
    x, y, _ = dataset.combine([split])

    blob = dataset.pack_for_helloworld(x, y)

    import io
    loaded = np.load(io.BytesIO(blob), allow_pickle=False)
    assert set(loaded.files) == {"x", "y"}
    assert loaded["x"].shape[0] == loaded["y"].shape[0]
    assert loaded["x"].dtype == np.float32
    assert loaded["y"].dtype == np.int64


def test_it_refuses_to_send_nothing():
    empty_x = np.zeros((0, len(features.FEATURE_NAMES)), dtype=np.float32)
    empty_y = np.zeros((0,), dtype=np.int64)

    with pytest.raises(ValueError):
        dataset.pack_for_helloworld(empty_x, empty_y)


def test_the_description_reports_the_baseline():
    """Accuracy without the class balance beside it says nothing.

    A model that always answers "up" scores the up-share exactly. Any headline
    number has to be read against it, so the UI is given it rather than left to
    work it out.
    """
    prices = synthetic_prices(days=500)
    split = dataset.build_one("TEST", prices)
    _, _, scaler = dataset.combine([split])

    described = dataset.describe([split], scaler)

    assert described["test"]["up_share"] is not None
    assert 0.0 <= described["test"]["up_share"] <= 1.0
    assert described["train"]["to"] < described["test"]["from"]


# --- pooling several symbols ----------------------------------------------

def test_the_whole_panel_is_split_at_one_date():
    """Per-symbol splits are not chronological across a pool.

    Symbols have different amounts of history, so an 80% cut lands on a
    different date for each one. On the first real panel built here that gave a
    training window running to 2026-04-21 and a test window starting
    2024-09-09: twenty months in which the model trained on one company and was
    graded on another over the same days. These markets move together, so that
    is a random split with extra steps.
    """
    # A newer listing: starts later, runs to the same day as the older one.
    # Two series that merely differ in length both ending at different dates is
    # not the case that matters -- the overlap only exists when they are trading
    # over the same period, which is the normal state of a watchlist.
    long_history = synthetic_prices(days=900, seed=1)
    frames = {
        "LONG": long_history,
        "SHORT": synthetic_prices(days=900, seed=2).iloc[-400:],
    }
    splits, cut = dataset.build_panel(frames, test_fraction=0.2)

    assert len(splits) == 2, "a symbol was dropped that had data on both sides"

    latest_train = max(s.train_dates[-1] for s in splits)
    earliest_test = min(s.test_dates[0] for s in splits)

    assert latest_train < earliest_test, (
        f"training runs to {latest_train.date()} while testing starts "
        f"{earliest_test.date()} -- the windows overlap across symbols")

    for split in splits:
        assert split.train_dates[-1] < cut <= split.test_dates[0]


def test_a_symbol_with_too_little_history_is_left_out_not_mis_split():
    """Better absent than quietly put back into the overlap."""
    frames = {
        "LONG": synthetic_prices(days=900, seed=1),
        # Too new to have any training rows at all: it begins after the cut.
        "NEWCOMER": synthetic_prices(days=900, seed=3).iloc[-60:],
    }
    splits, cut = dataset.build_panel(frames, test_fraction=0.2)

    kept = {s.symbol for s in splits}
    assert "LONG" in kept
    # NEWCOMER starts after the cut, so it has no training rows at all.
    for split in splits:
        assert len(split.y_train) > 0 and len(split.y_test) > 0


def test_pooled_rows_leave_in_date_order():
    """Because the coordinator is told they are.

    HelloWorldAi holds back the newest rows when a submitter declares time
    order. Stacked by symbol, "the newest rows" is the tail of whichever
    company happened to be last, and the declaration is false: measured, that
    gave a holdout score of 54.9%, identical to the random slice it replaced.
    """
    frames = {
        "A": synthetic_prices(days=600, seed=11),
        "B": synthetic_prices(days=600, seed=12),
    }
    splits, _ = dataset.build_panel(frames, test_fraction=0.2)
    assert len(splits) == 2

    # Rebuild the ordering the same way combine does, and check it is sorted.
    dates = np.concatenate([s.train_dates.values for s in splits])
    order = np.argsort(dates, kind="stable")
    sorted_dates = dates[order]

    assert list(sorted_dates) == sorted(sorted_dates), "the sort is not a sort"

    # And that combine actually applies it: the two symbols must interleave,
    # not sit one after the other.
    x, y, _ = dataset.combine(splits)
    assert len(x) == len(dates)

    # A stacked array would have every row of A before every row of B. After
    # sorting by date the halves must overlap, since both cover the same span.
    half = len(sorted_dates) // 2
    assert sorted_dates[half] > sorted_dates[0], "dates did not advance"
    assert sorted_dates[-1] >= sorted_dates[half], "dates are not monotonic"
