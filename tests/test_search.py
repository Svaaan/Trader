"""The three-way split, the trial ledger, and the seal on the test set.

Compute stopped being the constraint a while ago: a model trains in six seconds
and a card trains sixty-four at once. What is scarce now is the test set, and it
is scarce in a way that is easy to spend without noticing -- every configuration
scored against it is a question asked, and after a few hundred questions the best
answer is noise with a good hat on.

So these check the things that make a search survivable: that the search can
never see the sealed period, that the number of attempts is recorded, that the
bar rises with the count, and that opening the test set requires having decided
what you were testing first.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from conftest import synthetic_panel                          # noqa: E402
from trader import dataset, search                            # noqa: E402


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    monkeypatch.setattr(search, "SEARCH_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def prepared(offline_spec):
    return dataset.prepare(synthetic_panel(), offline_spec)


# --- the split --------------------------------------------------------------

def test_the_three_periods_run_in_order(prepared):
    cut = search.three_way_cut(prepared)
    assert cut.validation_start < cut.test_start


def test_the_search_can_never_see_the_sealed_period(prepared, offline_spec):
    """The whole point. A validation score computed on rows that are also in
    the test set would make every subsequent number meaningless."""
    cut = search.three_way_cut(prepared)

    splits, _ = dataset.split_at(prepared, cut.validation_start)
    windows = [search._truncate_before(s, cut.test_start) for s in splits]

    for window in windows:
        if not len(window.test_dates):
            continue
        assert window.test_dates.max() < cut.test_start, (
            "a validation row landed inside the sealed period")
        assert window.train_dates.max() < cut.validation_start


def test_the_test_boundary_matches_an_ordinary_run(prepared, offline_spec):
    """A sealed run and a normal one have to be graded on the same rows, or the
    sealed number cannot be compared to anything the project already has."""
    cut = search.three_way_cut(prepared, test_fraction=0.2)
    ordinary = dataset.choose_cut_date(prepared.assembled, prepared.label_frame,
                                       test_fraction=0.2)
    assert cut.test_start == ordinary


def test_training_rows_shrink_when_validation_is_carved_out(prepared):
    """Validation is taken out of training, not out of the test set."""
    cut = search.three_way_cut(prepared)

    search_splits, _ = dataset.split_at(prepared, cut.validation_start)
    final_splits, _ = dataset.split_at(prepared, cut.test_start)

    searching = sum(len(s.y_train) for s in search_splits)
    final = sum(len(s.y_train) for s in final_splits)
    assert searching < final, (
        "the search trained on as much as the committed model, so validation "
        "came from somewhere it should not have")


# --- the ledger -------------------------------------------------------------

def test_every_trial_is_counted(sealed, offline_spec):
    assert search.count_trials() == 0
    for index in range(3):
        search.record_trial(offline_spec, {"accuracy": 0.5 + index / 100,
                                           "edge": index / 100})
    assert search.count_trials() == 3
    assert [t["trial"] for t in search.read_trials()] == [1, 2, 3]


def test_a_trial_never_records_a_test_score(sealed, offline_spec):
    """A trial that wrote a test number would be a trial that had opened the
    test set -- which is the one thing the ledger exists to make visible."""
    entry = search.record_trial(offline_spec,
                                {"accuracy": 0.55, "edge": 0.02,
                                 "executable_sharpe": 1.2})
    assert set(entry["validation"]) <= {
        "accuracy", "baseline", "edge", "executable_sharpe",
        "executable_tstat", "rows", "effective_rows", "edge_standard_error",
        "edge_comparable", "days", "final_equity", "fees", "trades",
        "rebalances", "random_percentile", "random_median", "hold_median"}
    assert "test" not in json.dumps(entry).lower().replace("test_fraction", "")


def test_the_ledger_is_append_only(sealed, offline_spec):
    search.record_trial(offline_spec, {"accuracy": 0.5})
    path, _ = search._paths()
    before = open(path, encoding="utf-8").read()

    search.record_trial(offline_spec, {"accuracy": 0.6})
    assert open(path, encoding="utf-8").read().startswith(before)


def test_the_leaderboard_ranks_on_validation(sealed, offline_spec):
    for sharpe in (0.2, 1.9, 0.8):
        search.record_trial(offline_spec, {"accuracy": 0.5,
                                           "executable_sharpe": sharpe})
    best = search.leaderboard(3)
    assert [t["validation"]["executable_sharpe"] for t in best] == [1.9, 0.8, 0.2]


# --- what looking many times costs ------------------------------------------

def test_the_bar_rises_with_the_number_of_attempts():
    once = search.corrected_threshold(1, baseline=0.5, effective_rows=25_000)
    many = search.corrected_threshold(400, baseline=0.5, effective_rows=25_000)

    assert many["corrected"] > once["corrected"]
    assert many["inflation"] > 1.5
    # And the uncorrected risk over that many tries is near certainty.
    assert many["family_wise_risk_uncorrected"] > 0.99


def test_one_trial_is_the_ordinary_two_standard_errors():
    out = search.corrected_threshold(1, baseline=0.5, effective_rows=10_000)
    assert out["corrected"] == pytest.approx(out["naive"], rel=1e-3)
    assert out["inflation"] == pytest.approx(1.0, abs=1e-3)


def test_more_evidence_lowers_the_bar_at_a_fixed_trial_count():
    thin = search.corrected_threshold(100, baseline=0.5, effective_rows=1_000)
    thick = search.corrected_threshold(100, baseline=0.5, effective_rows=100_000)
    assert thick["corrected"] < thin["corrected"]


# --- the seal ---------------------------------------------------------------

def test_the_test_set_will_not_open_without_a_commitment(sealed, prepared):
    cut = search.three_way_cut(prepared)
    with pytest.raises(RuntimeError, match="Nothing has been committed"):
        search.open_test_set(prepared, cut)


def test_committing_records_the_reason_and_the_trial_count(sealed, offline_spec):
    for _ in range(7):
        search.record_trial(offline_spec, {"accuracy": 0.5})

    committed = search.commit(offline_spec, why="best validation Sharpe")
    assert committed["after_trials"] == 7
    assert committed["why"] == "best validation Sharpe"
    assert search.seal_state()["committed"]["spec"]["target"] == offline_spec.target


def test_opening_is_counted_and_the_second_one_says_so(sealed, prepared,
                                                       offline_spec):
    cut = search.three_way_cut(prepared)
    search.commit(offline_spec, why="testing")

    first = search.open_test_set(prepared, cut)
    assert first["opening_number"] == 1
    assert "opening number" not in first["reading"]

    second = search.open_test_set(prepared, cut)
    assert second["opening_number"] == 2
    assert "opening number 2" in second["reading"]
    assert "slower search" in second["reading"]
    assert len(search.seal_state()["opened"]) == 2


def test_the_reading_names_the_number_of_trials(sealed, prepared, offline_spec):
    for _ in range(50):
        search.record_trial(offline_spec, {"accuracy": 0.5})
    search.commit(offline_spec, why="testing")

    opening = search.open_test_set(prepared, search.three_way_cut(prepared))
    assert "Best of 50 configurations" in opening["reading"]
    assert opening["correction"]["trials"] == 50


def test_a_sealed_run_scores_on_the_same_rows_a_normal_run_would(
        sealed, prepared, offline_spec):
    cut = search.three_way_cut(prepared)
    search.commit(offline_spec, why="testing")
    opening = search.open_test_set(prepared, cut)

    splits, _ = dataset.split_at(prepared, cut.test_start)
    expected = sum(len(s.y_test) for s in splits)
    assert opening["scores"]["rows"] == expected


# --- running one ------------------------------------------------------------

def test_a_grid_is_every_combination():
    combos = search.grid(use_macro=[True, False], horizon=[1, 5, 20])
    assert len(combos) == 6
    assert {"use_macro": True, "horizon": 20} in combos


def test_a_search_records_one_trial_per_configuration(sealed, offline_spec):
    panel = synthetic_panel()
    combos = search.grid(use_cross=[True, False])

    out = search.run_search(panel, combos, base=offline_spec)

    assert out["ran"] == 2
    assert search.count_trials() == 2
    assert out["best"]


def test_a_search_never_opens_the_seal(sealed, offline_spec):
    panel = synthetic_panel()
    search.run_search(panel, search.grid(use_cross=[True, False]),
                      base=offline_spec)

    state = search.seal_state()
    assert state["committed"] is None
    assert state["opened"] == []


# --- the horizon survives every cut -------------------------------------------

def test_the_horizon_survives_the_split_and_the_truncation(offline_spec):
    """A Split rebuilt field by field would drop back to a one-day horizon, and
    every validation score at five days would be annualised as though it were
    daily -- which is precisely the slope a search would climb."""
    import dataclasses

    from trader import baseline

    spec = dataclasses.replace(offline_spec, horizon=5)
    prepared = dataset.prepare(synthetic_panel(), spec)
    cut = search.three_way_cut(prepared)

    splits, _ = dataset.split_at(prepared, cut.validation_start)
    assert {s.horizon for s in splits} == {5}

    truncated = [search._truncate_before(s, cut.test_start) for s in splits]
    assert {s.horizon for s in truncated} == {5}
    assert {s.horizon for s in (baseline._truncate(s, cut.test_start)
                                for s in splits)} == {5}

    _, _, scaler = dataset.combine(truncated, prepared.feature_names)
    assert dataset.test_matrix(truncated, scaler).horizon == 5



def test_a_measured_standard_error_sets_the_search_bar():
    """The same repair as the gate: the bar is sized by how the edge itself
    varies, and the single-proportion formula is only a fallback for trials
    recorded before that was measured."""
    measured = search.corrected_threshold(1, baseline=0.5, effective_rows=10_000,
                                          standard_error=0.01)
    assert measured["naive"] == pytest.approx(0.0196, abs=1e-4)

    fallback = search.corrected_threshold(1, baseline=0.5, effective_rows=10_000)
    assert fallback["naive"] == pytest.approx(1.96 * 0.005, abs=1e-4)


# --- the book: what is traded, as opposed to what is built ---------------------
#
# The audit of the sealed period found one slice whose edge was not already gone
# by the opening bell: long-only, high threshold. Testing that is a search over
# trading rules rather than over datasets -- same model, same rows, different
# subset held -- so it is its own axis, and each one costs a trial.

def test_a_long_only_book_holds_nothing_short(prepared):
    cut = search.three_way_cut(prepared)
    scores = search._score_on(prepared, cut, prepared.spec, period="validation",
                              book={"long_only": True, "min_probability": 0.52})

    assert scores["up_rate"] == 1.0
    assert scores["book"]["long_only"] is True
    assert scores["book_rows"] < scores.get("rows", 0) + scores["book_rows"]


def test_a_threshold_holds_fewer_rows_than_everything(prepared):
    cut = search.three_way_cut(prepared)
    everything = search._score_on(prepared, cut, prepared.spec,
                                  period="validation")
    picky = search._score_on(prepared, cut, prepared.spec, period="validation",
                             book={"min_probability": 0.52})

    assert picky["rows"] < everything["rows"]
    assert everything["book"] == search.DEFAULT_BOOK


def test_a_book_too_small_to_be_a_strategy_is_refused(prepared):
    cut = search.three_way_cut(prepared)
    with pytest.raises(ValueError, match="not a strategy"):
        search._score_on(prepared, cut, prepared.spec, period="validation",
                         book={"min_probability": 0.999})


def test_top_n_keeps_that_many_a_session(prepared):
    cut = search.three_way_cut(prepared)
    scores = search._score_on(prepared, cut, prepared.spec, period="validation",
                              book={"top_n": 2})

    assert scores["rows"] <= scores["days"] * 2


def test_the_book_is_recorded_with_the_trial(sealed, offline_spec):
    book = {"long_only": True, "min_probability": 0.58, "top_n": None}
    entry = search.record_trial(offline_spec, {"accuracy": 0.51}, book=book)

    assert entry["book"]["long_only"] is True
    assert entry["book"]["min_probability"] == 0.58
    assert search.read_trials()[0]["book"] == entry["book"]


def test_a_trial_without_a_book_records_the_default(sealed, offline_spec):
    entry = search.record_trial(offline_spec, {"accuracy": 0.51})
    assert entry["book"] == search.DEFAULT_BOOK


def test_the_seal_scores_the_book_that_was_committed(sealed, prepared,
                                                     offline_spec):
    """Committing a configuration without its trading rule would test something
    nobody chose."""
    book = {"long_only": True, "min_probability": 0.52, "top_n": None}
    search.commit(prepared.spec, why="the one slice that is reachable", book=book)

    opening = search.open_test_set(prepared, search.three_way_cut(prepared))
    assert opening["scores"]["book"]["long_only"] is True
    assert opening["scores"]["up_rate"] == 1.0
    assert search.seal_state()["committed"]["book"] == {**search.DEFAULT_BOOK,
                                                        **book}


def test_a_search_over_books_runs_one_trial_each(sealed, offline_spec):
    panel = synthetic_panel()
    out = search.run_search(panel, [{}], base=offline_spec,
                            books=[{"long_only": True, "min_probability": 0.52},
                                   {"min_probability": 0.52}])

    assert out["ran"] == 2
    assert search.count_trials() == 2
    assert {t["book"]["long_only"] for t in search.read_trials()} == {True, False}


def test_a_one_sided_book_has_no_edge_to_compare(prepared):
    """Long only means every call is up, and the baseline is always up, so the
    difference is identically zero and clears any bar trivially."""
    cut = search.three_way_cut(prepared)
    long_only = search._score_on(prepared, cut, prepared.spec,
                                 period="validation",
                                 book={"long_only": True, "min_probability": 0.52})
    both = search._score_on(prepared, cut, prepared.spec, period="validation")

    assert long_only["edge_comparable"] is False
    assert long_only["edge"] == 0.0
    assert both["edge_comparable"] is True


# --- a book scored in money ----------------------------------------------------
#
# A minimum fee is not a percentage: the same strategy is free at one account
# size and fatal at another. Measured on the real panel, the model's top five
# rebalanced every session paid 504 of commission on a 500 account and ended at
# zero; the same five every sixty sessions paid 92 and ended at 792.

def money_panel(sessions=120, names=8, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=sessions)
    columns = [f"S{i}" for i in range(names)]
    return pd.DataFrame(rng.normal(0.0005, 0.01, size=(sessions, names)),
                        index=dates, columns=columns)


def test_carrying_a_position_costs_nothing_and_trading_it_costs_real_money():
    from trader import book

    moves = money_panel()
    held = [{"S0": 1.0} for _ in range(len(moves))]
    churned = [{"S0" if i % 2 else "S1": 1.0} for i in range(len(moves))]

    quiet = book.simulate(held, moves, account=500.0)
    busy = book.simulate(churned, moves, account=500.0)

    assert quiet["trades"] == 2                      # in at the start, out at the end
    assert busy["trades"] > 100
    assert busy["fees"] > 20 * quiet["fees"]


def test_an_account_that_runs_out_of_money_stops():
    from trader import book

    moves = money_panel(sessions=400)
    # Twenty names rebalanced every session on a tiny account: the minimum fee
    # alone is more than the account holds.
    churn = []
    for index in range(len(moves)):
        picked = list(moves.columns[(index % 2)::2])
        churn.append({name: 1.0 / len(picked) for name in picked})

    out = book.simulate(churn, moves, account=50.0)
    assert out["broke"] is True
    assert out["final"] == 0.0


def test_the_rebalance_interval_decides_how_often_it_trades():
    from trader import book

    moves = money_panel(sessions=120)
    signal = pd.DataFrame([
        {"date": date, "symbol": name, "p": 0.5 + (index % 7) / 20}
        for index, date in enumerate(moves.index) for name in moves.columns])

    daily = book.targets(signal, list(moves.index), top=2, every=1)
    monthly = book.targets(signal, list(moves.index), top=2, every=20)

    assert len(monthly) == len(daily) == len(moves)
    # Between rebalances the book is simply held.
    assert monthly[5] == monthly[1]
    assert (book.simulate(monthly, moves)["trades"]
            < book.simulate(daily, moves)["trades"])


def test_the_percentile_is_measured_against_the_same_trade_pattern():
    from trader import book

    moves = money_panel()
    against = book.percentile(1e9, moves, top=3, every=20, draws=25, seed=1)
    assert against["beaten"] == 1.0
    assert against["draws"] == 25
    assert against["worst_tenth"] <= against["median"] <= against["best_tenth"]

    hopeless = book.percentile(0.0, moves, top=3, every=20, draws=25, seed=1)
    assert hopeless["beaten"] == 0.0


def test_a_money_trial_records_what_the_account_did(sealed, offline_spec):
    panel = synthetic_panel()
    out = search.run_search(panel, [{}], base=offline_spec,
                            books=[{"long_only": True, "top_n": 2,
                                    "rebalance_every": 20, "account": 500.0}])

    assert out["ran"] == 1
    recorded = search.read_trials()[0]["validation"]
    assert recorded["final_equity"] is not None
    assert recorded["trades"] > 0
    assert 0.0 <= recorded["random_percentile"] <= 1.0
    assert recorded["hold_median"] is not None
    # An account cannot be scored on an accuracy it never had a baseline for.
    assert recorded["edge_comparable"] is False
