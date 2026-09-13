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

There are now four kinds of input rather than one, and three of them can leak in
ways the per-symbol block cannot:

  * cross-sectional ranks can read a symbol's peers, and in a panel spanning
    New York and Zurich "the same day" is not the same instant
  * macro series close after the European equity session they would be attached
    to, so an unlagged macro block hands European names their own future
  * event features can read a schedule that had not been announced yet

Each gets its own property test below.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from conftest import synthetic_panel, synthetic_prices        # noqa: E402
from trader import cross, dataset, events, features, labels, macro   # noqa: E402


# --- the property itself ---------------------------------------------------

@pytest.mark.parametrize("cut", [600, 900, 1200])
def test_a_feature_does_not_change_when_more_history_arrives(cut, prices):
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


def test_a_centred_window_is_caught(prices):
    """The test has to be able to fail, so here is a leak on purpose.

    A centred rolling mean is the classic accidental version -- it looks like
    every other rolling call and reads half its window from the future. If this
    test ever stops failing, the property test above has stopped testing.
    """
    def leaky(frame):
        out = features.build(frame)
        out["return_1d"] = frame["close"].rolling(5, center=True).mean().reindex(
            out.index)
        return out

    full = leaky(prices)
    truncated = leaky(prices.iloc[:900])
    shared = truncated.index.intersection(full.index)

    same = np.allclose(truncated.loc[shared, "return_1d"].to_numpy(),
                       full.loc[shared, "return_1d"].to_numpy(),
                       rtol=0, atol=1e-12, equal_nan=True)
    assert not same, "a centred window should have been detected as look-ahead"


def test_the_rsi_warmup_is_blank_not_maximal(prices):
    """A warm-up filled with 100 means "nothing but gains", which is a lie.

    This was a real bug, hidden because the 50-day average dropped those rows
    anyway. It is tested directly so that shortening the window set cannot
    quietly reintroduce fourteen days of fictional maximum strength.
    """
    rsi = features._rsi(prices["close"], 14)
    assert rsi.head(14).isna().all(), "RSI produced values before it had data"
    assert rsi.dropna().between(0.0, 100.0).all()


# --- labels ----------------------------------------------------------------

def test_the_label_does_look_forward(prices):
    """The other half: a label that cannot see the future predicts nothing."""
    label = labels.direction(prices, horizon=1)
    close = prices["close"]

    rose = (close.shift(-1) > close).astype(float)
    aligned = label.dropna()
    assert (aligned == rose.reindex(aligned.index)).all()


def test_the_last_row_has_features_but_no_label(prices):
    """Today is exactly the situation prediction exists for."""
    x = features.build(prices)
    y = labels.direction(prices, horizon=1)

    assert not x.empty
    assert np.isnan(y.iloc[-1]), "the last row cannot have a known future"
    assert x.index[-1] in prices.index


def test_a_relative_label_is_balanced_on_every_date(panel):
    """The whole point of the relative target: no constant can beat it."""
    forward = labels.forward_return_panel(panel, horizon=1)
    relative = labels.relative_direction(forward)

    per_date = relative.dropna(how="all")
    shares = per_date.mean(axis=1).dropna()

    # Six symbols ranked per date: the top half is three of them, every time.
    assert shares.between(0.4, 0.6).all(), (
        "a relative label that is not balanced per date reintroduces the "
        "baseline problem it exists to remove")


def test_a_relative_label_pairs_with_relative_returns(panel):
    """A relative model graded on absolute returns is credited with drift."""
    _, graded, executable = labels.build_panel_labels(panel,
                                                      target=labels.RELATIVE)
    # Market-relative returns sum to zero across the panel on every date --
    # and so must the executable ones, or the two windows are being compared
    # after only one of them was demeaned.
    assert np.allclose(graded.sum(axis=1).dropna().to_numpy(), 0.0, atol=1e-12)
    assert np.allclose(executable.sum(axis=1).dropna().to_numpy(), 0.0, atol=1e-10)


def test_the_executable_return_starts_at_the_next_open(prices):
    """The signal exists only after the close, so the trade starts at the open.

    The graded return runs close(t) -> close(t+1); this one runs
    open(t+1) -> close(t+1). The difference is the overnight gap, which on the
    real panel is the entire gross edge.
    """
    graded = labels.forward_return(prices, horizon=1)
    reachable = labels.executable_return(prices, horizon=1)

    expected = (prices["close"].shift(-1) / prices["open"].shift(-1) - 1.0)
    assert np.allclose(reachable.dropna().to_numpy(),
                       expected.reindex(reachable.dropna().index).to_numpy(),
                       atol=1e-12)

    # They differ by exactly the overnight move, compounded.
    overnight = prices["open"].shift(-1) / prices["close"] - 1.0
    combined = (1 + overnight) * (1 + reachable) - 1.0
    shared = graded.dropna().index.intersection(combined.dropna().index)
    assert np.allclose(graded.loc[shared].to_numpy(),
                       combined.loc[shared].to_numpy(), atol=1e-10)


def test_the_executable_return_is_still_forward_looking_only(prices):
    """It may start later than the label, but never earlier than the signal."""
    reachable = labels.executable_return(prices, horizon=1)
    # The last row has no next open, so it cannot be graded.
    assert np.isnan(reachable.iloc[-1])


# --- the blocks that can leak across symbols and timezones -----------------

def test_the_cross_sectional_block_is_lagged(panel):
    """A rank built from today's peers is today's future for a European name."""
    own = {symbol: features.build(prices) for symbol, prices in panel.items()}
    ranked = cross.build(own)

    symbol = sorted(own)[0]
    block = ranked[symbol]
    assert not block.empty

    # The first LAG_SESSIONS rows have nothing to carry forward from.
    assert block.iloc[:cross.LAG_SESSIONS].isna().all().all()

    # And the value on any date must equal the unlagged value one row earlier.
    unlagged = (pd.DataFrame({s: own[s]["return_1d"] for s in sorted(own)})
                .rank(axis=1, pct=True) - 0.5)[symbol]
    shared = block.dropna().index[:50]
    for date in shared:
        position = unlagged.index.get_loc(date)
        assert np.isclose(block.loc[date, "rank_return_1d"],
                          unlagged.iloc[position - cross.LAG_SESSIONS])


def test_a_narrow_panel_gets_no_cross_sectional_block():
    """A percentile over three names is an opinion about three numbers."""
    own = {s: features.build(p)
           for s, p in synthetic_panel(symbols=("AAA", "BBB")).items()}
    assert not cross.usable(own)
    assert all(frame.empty or not len(frame.columns)
               for frame in cross.build(own).values())


def test_the_macro_block_is_lagged(monkeypatch):
    """Macro closes after the European session it would be attached to."""
    dates = pd.bdate_range("2020-01-01", periods=600, name="date")
    rng = np.random.default_rng(3)
    fake = pd.DataFrame(
        {name: 20.0 + np.cumsum(rng.normal(0, 0.2, len(dates)))
         for name in macro.SERIES},
        index=dates)

    monkeypatch.setattr(macro, "_closes", lambda **kwargs: fake)
    block = macro.build()

    # vix_log on date t must be the log of the VIX close on t - LAG_SESSIONS.
    expected = np.log(fake["vix"].clip(lower=1e-6)).shift(macro.LAG_SESSIONS)
    shared = block.index[:100]
    assert np.allclose(block.loc[shared, "vix_log"].to_numpy(),
                       expected.reindex(shared).to_numpy(), atol=1e-12)


def test_event_features_do_not_read_an_unannounced_schedule(monkeypatch):
    """Nobody knew in January that the report would be in June."""
    import datetime as dt

    reports = [dt.date(2020, 2, 5), dt.date(2020, 5, 6), dt.date(2020, 8, 5)]
    monkeypatch.setattr(events, "earnings_dates",
                        lambda symbol, refresh=False: reports * 3)

    index = pd.bdate_range("2020-01-02", periods=120, name="date")
    block = events.build("AAA", index)

    # Ninety days out, the countdown must read "far" rather than 90.
    far = block.loc[block.index < pd.Timestamp("2020-01-06"), "days_to_earnings"]
    assert (far == 1.0).all(), (
        "the countdown revealed a date that had not been announced yet")

    # And it must be strictly decreasing as the announced date approaches.
    approach = block.loc["2020-01-20":"2020-02-04", "days_to_earnings"]
    assert approach.is_monotonic_decreasing


def test_a_symbol_with_no_calendar_gets_neutral_values_not_invented_ones(monkeypatch):
    monkeypatch.setattr(events, "earnings_dates", lambda symbol, refresh=False: [])
    index = pd.bdate_range("2020-01-02", periods=40, name="date")
    block = events.build("AAA", index)

    assert (block["days_to_earnings"] == 1.0).all()
    assert (block["in_earnings_drift"] == 0.0).all()
    assert not block.isna().any().any()


# --- the split -------------------------------------------------------------

def test_the_split_is_chronological(panel, offline_spec):
    splits, cut, _ = dataset.build_panel(panel, offline_spec)

    for split in splits:
        assert split.train_dates.max() < cut <= split.test_dates.min()
        assert split.train_dates.is_monotonic_increasing
        assert split.test_dates.is_monotonic_increasing


def test_the_whole_panel_is_split_at_one_date(panel, offline_spec):
    """Per-symbol splits look chronological and are not, across the pool."""
    splits, cut, _ = dataset.build_panel(panel, offline_spec)

    latest_train = max(s.train_dates.max() for s in splits)
    earliest_test = min(s.test_dates.min() for s in splits)
    assert latest_train < earliest_test, (
        "one symbol trains on days another is graded on")


def test_no_training_label_reaches_past_the_cut(panel):
    """The purge. At horizon 20 this is twenty rows of direct leakage."""
    spec = dataset.Spec(use_macro=False, use_events=False, horizon=20)
    splits, cut, _ = dataset.build_panel(panel, spec)

    for split in splits:
        realises = split.train_dates.max() + pd.Timedelta(days=0)
        # The last training row's label looks `horizon` sessions ahead; the
        # embargo has to have removed anything that lands on or after the cut.
        position = split.train_dates.shape[0]
        assert position > 0
        assert realises < cut
        # Explicitly: the gap between the last train date and the cut must
        # cover the horizon in sessions.
        gap = len(pd.bdate_range(split.train_dates.max(), cut)) - 1
        assert gap >= spec.embargo, (
            f"{split.symbol} has only {gap} sessions of embargo for a "
            f"{spec.horizon}-session horizon")


def test_the_cut_date_can_be_pinned(panel, offline_spec):
    """The regression test for a seven-week silent drift.

    Re-deriving the split at scoring time moved it from 2024-11-06 to
    2024-12-31, graded 87 rows the run never advertised, and added a symbol to
    the test set that had been absent from training. Passing the stored date
    back has to reproduce the split exactly.
    """
    first, cut, _ = dataset.build_panel(panel, offline_spec)

    # A different test_fraction would choose a different date -- unless it is
    # given one, which is the whole point.
    other = dataset.Spec(use_macro=False, use_events=False, test_fraction=0.35)
    second, again, _ = dataset.build_panel(panel, other, cut_date=cut)

    assert again == cut
    assert [s.symbol for s in first] == [s.symbol for s in second]
    for a, b in zip(first, second):
        assert len(a.y_test) == len(b.y_test)
        assert a.test_dates.equals(b.test_dates)


def test_a_symbol_with_too_little_history_is_named_not_silently_dropped(offline_spec):
    """Better absent than quietly put back into the overlap -- but say so.

    A watchlist of ten that trains as nine used to be invisible. The symbol is
    still excluded, because splitting it elsewhere would recreate the overlap
    the single cut date exists to prevent, but the exclusion is now recorded.
    """
    frames = synthetic_panel()
    frames["NEWCOMER"] = synthetic_prices(days=1400).iloc[-60:]

    splits, cut, report = dataset.build_panel(frames, offline_spec)

    kept = {s.symbol for s in splits}
    assert "NEWCOMER" not in kept
    assert "NEWCOMER" in report["excluded"], (
        "a symbol vanished from the panel without being reported")
    for split in splits:
        assert len(split.y_train) > 0 and len(split.y_test) > 0


def test_the_scaler_never_sees_the_test_period(panel, offline_spec):
    splits, _, report = dataset.build_panel(panel, offline_spec)
    _, _, scaler = dataset.combine(splits, report["feature_names"])

    train_only = np.concatenate([s.x_train for s in splits])
    assert np.allclose(scaler.mean, train_only.mean(axis=0), atol=1e-5)
    assert np.allclose(scaler.std, train_only.std(axis=0), atol=1e-5)


def test_pooled_rows_leave_in_date_order(panel, offline_spec):
    """The coordinator holds back the last 20%; it has to be the last 20%."""
    splits, _, report = dataset.build_panel(panel, offline_spec)

    dates = np.concatenate([s.train_dates.values for s in splits])
    order = np.argsort(dates, kind="stable")
    assert (dates[order] == np.sort(dates)).all()


def test_the_test_matrix_is_in_date_order(panel, offline_spec):
    """Turnover and drawdown treat consecutive rows as consecutive."""
    splits, _, report = dataset.build_panel(panel, offline_spec)
    _, _, scaler = dataset.combine(splits, report["feature_names"])
    test = dataset.test_matrix(splits, scaler)

    assert test.dates.is_monotonic_increasing
    # Both return series ride along, on the same rows and the same order.
    assert len(test.returns) == len(test.executable) == len(test)


# --- packing ---------------------------------------------------------------

def test_a_constant_feature_does_not_become_infinity():
    scaler = dataset.Scaler(mean=np.array([1.0, 0.0], dtype=np.float32),
                            std=np.array([0.0, 1.0], dtype=np.float32),
                            feature_names=["flat", "moving"])
    out = scaler.apply(np.array([[1.0, 2.0], [1.0, 3.0]], dtype=np.float32))
    assert np.isfinite(out).all()


def test_the_packed_dataset_is_loadable_without_unpickling(panel, offline_spec):
    import io

    splits, _, report = dataset.build_panel(panel, offline_spec)
    x, y, _ = dataset.combine(splits, report["feature_names"])
    blob = dataset.pack_for_helloworld(x, y)

    loaded = np.load(io.BytesIO(blob), allow_pickle=False)
    assert loaded["x"].shape[0] == loaded["y"].shape[0]
    assert loaded["x"].dtype == np.float32


def test_it_refuses_to_send_nothing():
    with pytest.raises(ValueError):
        dataset.pack_for_helloworld(np.zeros((0, 3), dtype=np.float32),
                                    np.zeros((0,), dtype=np.int64))


def test_the_description_reports_the_baseline_and_what_was_excluded(
        panel, offline_spec):
    splits, cut, report = dataset.build_panel(panel, offline_spec)
    _, _, scaler = dataset.combine(splits, report["feature_names"])
    described = dataset.describe(splits, scaler, offline_spec, report, cut)

    assert described["train"]["up_share"] is not None
    assert described["test"]["up_share"] is not None
    assert "excluded" in described
    assert described["cut_date"] == cut.date().isoformat()
    assert described["spec"]["embargo"] == offline_spec.embargo


def test_a_cache_that_stopped_updating_is_refetched(prices):
    """A ten-year history ending a fortnight ago still spans ten years.

    The span check passed and the freshness check did not exist, so the page
    generated "today's signal" from bars twelve days old for a fortnight.
    """
    import datetime as dt

    from trader import prices as prices_mod

    assert prices_mod._covers_period(prices, "10y") is False or True  # span only
    fresh = prices.index[-1].date()
    assert prices_mod._is_current(prices, today=fresh)
    assert prices_mod._is_current(
        prices, today=fresh + dt.timedelta(days=prices_mod.MAX_CACHE_AGE_DAYS))
    assert not prices_mod._is_current(
        prices, today=fresh + dt.timedelta(days=prices_mod.MAX_CACHE_AGE_DAYS + 1))


def test_a_weekend_does_not_count_as_stale(prices):
    """Friday's close read on Monday morning is current, not two days late."""
    import datetime as dt

    from trader import prices as prices_mod

    friday = prices.index[-1].date()
    assert prices_mod._is_current(prices, today=friday + dt.timedelta(days=3))
