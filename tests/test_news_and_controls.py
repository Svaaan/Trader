"""The news store's point-in-time discipline, and the local control models.

The store exists to answer one question honestly: what did this machine
provably know, and when. Everything below is about the ways that guarantee can
be lost -- by trusting a provider's timestamp, by rewriting history, or by using
a block that has no history yet and is therefore a date stamp wearing a
feature's name.

The controls are the other half of the review that produced this work: nine
models were trained on a GPU across a network and never compared against
anything but the class balance. These check that the comparison exists and that
it is made on the same rows.
"""

import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from conftest import synthetic_panel                          # noqa: E402
from trader import baseline, dataset, news                    # noqa: E402


@pytest.fixture
def store(tmp_path, monkeypatch):
    """An isolated news store, so tests never touch the real one."""
    monkeypatch.setattr(news, "STORE_DIR", str(tmp_path))
    return tmp_path


def fake_items(*ids):
    return [{"id": str(i), "title": f"headline {i}", "summary": "body",
             "published_utc": "2020-01-01T00:00:00Z", "provider": "test",
             "url": f"https://example.invalid/{i}"} for i in ids]


# --- the store --------------------------------------------------------------

def test_collecting_twice_adds_nothing_the_second_time(store, monkeypatch):
    monkeypatch.setattr(news, "_fetch", lambda symbol: fake_items(1, 2, 3))

    first = news.collect(["AAA"])
    second = news.collect(["AAA"])

    assert first["AAA"] == 3
    assert second["AAA"] == 0, "an idempotent pass duplicated items"


def test_the_store_is_append_only(store, monkeypatch):
    """Earlier lines must survive later passes untouched.

    The entire value of the file is that its contents could not have been
    adjusted after the fact. A store that gets rewritten is an archive again,
    and an archive is what cannot be trusted.
    """
    monkeypatch.setattr(news, "_fetch", lambda symbol: fake_items(1, 2))
    news.collect(["AAA"])
    path = news._store_path("AAA")
    before = open(path, encoding="utf-8").read()

    monkeypatch.setattr(news, "_fetch", lambda symbol: fake_items(3))
    news.collect(["AAA"])
    after = open(path, encoding="utf-8").read()

    assert after.startswith(before), "earlier lines were rewritten"
    assert len(after.splitlines()) == 3


def test_capture_time_is_written_by_us_not_taken_from_the_provider(store,
                                                                   monkeypatch):
    """A provider's `published_at` is frequently its ingestion time.

    Features count by capture time precisely because that is the one claim
    about an item that nobody downstream can revise.
    """
    monkeypatch.setattr(news, "_fetch", lambda symbol: fake_items(1))
    stamp = dt.datetime(2026, 5, 5, 12, 0, tzinfo=dt.timezone.utc)
    news.collect(["AAA"], now=stamp)

    stored = news._read("AAA")[0]
    assert stored["captured_utc"].startswith("2026-05-05")
    # The provider's claim is kept, clearly separate, and never measured from.
    assert stored["published_utc"] == "2020-01-01T00:00:00Z"


def test_features_count_by_capture_not_by_the_claimed_date(store, monkeypatch):
    """An item claiming last Tuesday but first seen today counts from today."""
    monkeypatch.setattr(news, "_fetch", lambda symbol: fake_items(1, 2))
    news.collect(["AAA"], now=dt.datetime(2024, 3, 15, tzinfo=dt.timezone.utc))

    index = pd.bdate_range("2024-03-01", "2024-03-29", name="date")
    block = news.build("AAA", index)

    # Nothing before capture, despite the items claiming 2020.
    early = block.loc[block.index < pd.Timestamp("2024-03-15"), "news_count_1d"]
    assert (early == 0.0).all(), "an item was counted before it was captured"
    # And something after, lagged by a session.
    assert block.loc["2024-03-18":, "news_count_5d"].max() > 0


def test_an_item_captured_today_is_not_available_to_today(store, monkeypatch):
    """The same one-session lag the macro and cross-sectional blocks take."""
    monkeypatch.setattr(news, "_fetch", lambda symbol: fake_items(1))
    news.collect(["AAA"], now=dt.datetime(2024, 3, 15, tzinfo=dt.timezone.utc))

    index = pd.bdate_range("2024-03-11", "2024-03-22", name="date")
    block = news.build("AAA", index)
    assert block.loc[pd.Timestamp("2024-03-15"), "news_count_1d"] == 0.0


def test_an_empty_store_is_not_ready_and_says_so(store):
    state = news.readiness(["AAA", "BBB"])
    assert state["items"] == 0
    assert not state["ready"]
    assert state["needed_days"] == news.MIN_HISTORY_DAYS


def test_a_young_store_is_not_ready(store, monkeypatch):
    """A block that is zeros for nine years and real for a week is a date stamp."""
    monkeypatch.setattr(news, "_fetch", lambda symbol: fake_items(1, 2))
    news.collect(["AAA"])
    state = news.readiness(["AAA"])
    assert state["items"] == 2
    assert not state["ready"], "a store collected today cannot be usable history"


def test_an_unparseable_line_does_not_cost_the_whole_history(store, monkeypatch):
    monkeypatch.setattr(news, "_fetch", lambda symbol: fake_items(1, 2))
    news.collect(["AAA"])
    with open(news._store_path("AAA"), "a", encoding="utf-8") as handle:
        handle.write("{not json\n")

    assert len(news._read("AAA")) == 2


def test_items_without_an_id_or_title_are_not_stored(store, monkeypatch):
    def messy(symbol):
        return [{"content": {"id": None, "title": "no id"}},
                {"content": {"id": "9", "title": None}},
                {"id": "10", "content": {"title": "fine", "summary": "s"}}]

    monkeypatch.setattr("yfinance.Ticker", lambda s: type(
        "T", (), {"news": messy(s)})())
    assert news.collect(["AAA"])["AAA"] == 1


# --- the controls -----------------------------------------------------------

@pytest.fixture
def splits_and_names(offline_spec):
    panel = synthetic_panel()
    splits, _, report = dataset.build_panel(panel, offline_spec)
    return splits, report["feature_names"]


def test_the_controls_run_on_the_same_rows_as_the_model(splits_and_names):
    splits, names = splits_and_names
    out = baseline.run_controls(splits, names, seeds=2)

    for control in ("majority", "logistic", "local_mlp"):
        assert control in out
        assert out[control]["rows"] == sum(len(s.y_test) for s in splits)
        # All three graded against the same, training-derived baseline.
        assert out[control]["baseline_accuracy"] == out["majority"]["accuracy"]


def test_the_majority_control_answers_one_way(splits_and_names):
    splits, names = splits_and_names
    out = baseline.run_controls(splits, names, seeds=1)
    assert out["majority"]["up_rate"] in (0.0, 1.0)
    assert out["majority"]["edge"] == 0.0


def test_the_noise_floor_is_the_spread_between_identical_configurations(
        splits_and_names):
    splits, names = splits_and_names
    out = baseline.run_controls(splits, names, seeds=3)

    floor = out["noise_floor"]
    assert floor["seeds"] == 3
    assert len(floor["accuracies"]) == 3
    assert floor["spread"] == pytest.approx(
        max(floor["accuracies"]) - min(floor["accuracies"]), abs=1e-9)
    assert floor["spread"] >= 0


def test_logistic_regression_separates_data_it_can_actually_separate():
    """A control that cannot learn a signal is not a control.

    If the logistic baseline could not find an obvious linear relationship, its
    failing to beat the class balance on real prices would say nothing.
    """
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 4))
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(int)

    model = baseline.fit_logistic(x, y)
    accuracy = ((model.probabilities(x) > 0.5) == y).mean()
    assert accuracy > 0.9


def test_the_local_mlp_learns_something_linear_models_cannot():
    """XOR, which is the point of having a non-linear control at all."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4000, 2))
    y = ((x[:, 0] > 0) ^ (x[:, 1] > 0)).astype(int)

    linear = baseline.fit_logistic(x, y)
    network = baseline.fit_mlp(x, y, hidden=32, depth=2, steps=4000, seed=0)

    assert ((linear.probabilities(x) > 0.5) == y).mean() < 0.6
    assert ((network.probabilities(x) > 0.5) == y).mean() > 0.9


def test_the_sigmoid_does_not_overflow_on_confident_scores():
    extreme = np.array([-2000.0, 0.0, 2000.0])
    out = baseline._sigmoid(extreme)
    assert np.isfinite(out).all()
    assert out[0] == pytest.approx(0.0)
    assert out[2] == pytest.approx(1.0)


def test_walk_forward_grades_several_consecutive_windows(offline_spec):
    panel = synthetic_panel()
    result = baseline.walk_forward(panel, offline_spec, folds=4)

    assert result["folds_run"] >= 3
    windows = [(f["test_from"], f["test_to"]) for f in result["folds"]]
    # Each window has to start after the one before it, or they are not folds.
    assert windows == sorted(windows)
    assert result["mean_edge"] is not None
    assert 0 <= result["folds_positive"] <= result["folds_run"]


def test_walk_forward_folds_do_not_train_on_their_own_test_window(offline_spec):
    panel = synthetic_panel()
    result = baseline.walk_forward(panel, offline_spec, folds=4)
    for fold in result["folds"]:
        assert fold["cut"] <= fold["test_from"]


# --- the block that can be asked for and refused ---------------------------

def test_the_news_block_is_refused_over_a_young_store(store, monkeypatch):
    """Asking for it is not the same as getting it.

    A block that is zeros for every historical row and a real number for today
    is not a feature, it is a date stamp -- and it would be learned as one.
    """
    monkeypatch.setattr(news, "_fetch", lambda symbol: fake_items(1, 2))
    news.collect(["AAA"])

    spec = dataset.Spec(use_macro=False, use_events=False, use_news=True)
    _, names, report = dataset.assemble(synthetic_panel(), spec)

    assert report["news"]["used"] is False
    assert not any(n in names for n in news.NEWS_NAMES), (
        "an unready news block reached the feature list")
    assert report["news"]["readiness"]["ready"] is False


def test_the_news_block_is_used_once_the_store_has_history(store, monkeypatch):
    monkeypatch.setattr(news, "_fetch", lambda symbol: fake_items(1, 2, 3))
    old = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=500)
    for symbol in ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF"):
        news.collect([symbol], now=old)

    spec = dataset.Spec(use_macro=False, use_events=False, use_news=True)
    assembled, names, report = dataset.assemble(synthetic_panel(), spec)

    assert report["news"]["used"] is True
    for column in news.NEWS_NAMES:
        assert column in names
    assert all(column in frame.columns
               for frame in assembled.values() for column in news.NEWS_NAMES)


def test_the_mlp_control_costs_the_same_on_a_wide_panel_as_a_narrow_one():
    """Counted in gradient steps, not epochs.

    Epoch-counting made the controls scale with the number of symbols, so a
    240-symbol panel cost twenty-four times a 10-symbol one -- minutes, which is
    long enough that somebody turns the controls off, which is how the project
    got into this state in the first place.
    """
    import time

    rng = np.random.default_rng(0)
    small = rng.normal(size=(2_000, 8))
    large = rng.normal(size=(60_000, 8))
    y_small = (small[:, 0] > 0).astype(int)
    y_large = (large[:, 0] > 0).astype(int)

    start = time.perf_counter()
    baseline.fit_mlp(small, y_small, steps=500, seed=0)
    narrow = time.perf_counter() - start

    start = time.perf_counter()
    baseline.fit_mlp(large, y_large, steps=500, seed=0)
    wide = time.perf_counter() - start

    # Thirty times the rows, same number of gradient steps, same work.
    assert wide < narrow * 4 + 0.5, (
        f"a 30x wider panel took {wide / max(narrow, 1e-9):.1f}x as long; "
        f"the control is counting epochs again")


def test_the_controls_record_the_hyperparameters_they_used():
    """A control trained differently from the model is not a control."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 4))
    y = (x[:, 0] > 0).astype(int)

    from trader import dataset as dataset_mod

    splits = [dataset_mod.Split(
        symbol="AAA", x_train=x[:400], y_train=y[:400],
        x_test=x[400:], y_test=y[400:],
        train_dates=pd.bdate_range("2020-01-01", periods=400),
        test_dates=pd.bdate_range("2021-08-01", periods=100),
        forward_returns_test=rng.normal(0, 0.01, 100))]

    out = baseline.run_controls(splits, [f"f{i}" for i in range(4)],
                                steps=250, seeds=1)
    assert out["hyperparameters"]["steps"] == 250
    assert out["hyperparameters"]["hidden"] == 64
