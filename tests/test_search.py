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
        "days"}
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
