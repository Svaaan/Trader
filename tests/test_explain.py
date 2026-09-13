"""The gate, and the reasons underneath it.

The most dangerous thing this project could do is print "Buy now" over a model
that has learned nothing. Every model it has trained so far is exactly that:
51.2% against a 51.6% baseline, no edge at all. A page that ranked those
probabilities anyway would be a random number generator with confident
typography, and it would be read by the person who built it, who would like it
to work.

So a call is gated on evidence, and every threshold is computed rather than
chosen. These check that the gate holds in both directions -- that it suppresses
when it should, and that it does not suppress a model that has genuinely earned
an opinion.

Three of the six hurdles exist because of things this project measured about
itself, and each has a test of its own below:

  * three identical configurations scored 51.44, 51.64 and 51.78, a spread
    wider than any edge ever measured -- so an edge smaller than the seed
    spread is a fact about the seed
  * a single test window is a single draw, and an edge that shows up in one of
    six consecutive windows is what a coin looks like
  * on the 238-symbol panel the logistic control clears every accuracy hurdle
    there is and loses 3.2% a year, so accuracy alone cannot open the gate
"""

import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trader import evaluate, explain                          # noqa: E402
from trader.dataset import Scaler                             # noqa: E402


def evaluation(accuracy, baseline, *, rows=4000, effective=2400, days=430,
               up_rate=0.5, tstat=3.0, sharpe=1.2):
    """An evaluation that passes every hurdle except the one under test.

    The return figures default to comfortably profitable so that a test about
    the accuracy hurdles is not silently also a test of the money hurdle. The
    money tests below override them.
    """
    return {"accuracy": accuracy, "baseline_accuracy": baseline,
            "rows": rows, "effective_rows": effective, "days": days,
            "up_rate": up_rate,
            "strategy_tstat": tstat, "strategy_sharpe": sharpe}


def controls(spread):
    return {"noise_floor": {"spread": spread}}


def folds(positive, run=6):
    return {"folds_positive": positive, "folds_run": run}


def named(trust, name):
    return next(c for c in trust.checks if c["name"] == name)


# --- the gate --------------------------------------------------------------

def test_no_edge_is_not_trusted():
    trust = explain.assess(evaluation(0.515, 0.516))
    assert not trust.trusted
    assert "no better" in trust.reason
    assert not named(trust, "beats_baseline")["passed"]


def test_an_edge_smaller_than_chance_is_not_trusted():
    # Half a point over 2,400 independent rows is inside the noise.
    trust = explain.assess(evaluation(0.521, 0.516))
    assert not trust.trusted
    assert not named(trust, "beats_chance")["passed"]


def test_a_clear_edge_is_trusted():
    trust = explain.assess(evaluation(0.56, 0.516),
                           controls=controls(0.004),
                           walk_forward=folds(5))
    assert trust.trusted
    assert all(check["passed"] for check in trust.checks)


def test_the_bar_rises_when_there_is_less_evidence():
    little = explain.assess(evaluation(0.60, 0.516, effective=300))
    lots = explain.assess(evaluation(0.60, 0.516, effective=20000))
    assert little.needed > lots.needed


def test_too_few_days_is_never_trusted_however_good_it_looks():
    """A wide panel over three weeks is still three weeks."""
    trust = explain.assess(evaluation(0.75, 0.51, rows=6000, effective=3000,
                                      days=30))
    assert not trust.trusted
    assert "distinct days" in trust.reason


def test_the_effective_sample_size_is_what_counts_not_the_row_count():
    """Ten symbols on one day are not ten independent verdicts.

    Same accuracy, same baseline, same row count -- only the discount for how
    much the panel moves together differs, and it decides the outcome.
    """
    honest = explain.assess(evaluation(0.532, 0.516, rows=4000, effective=800),
                            controls=controls(0.002), walk_forward=folds(6))
    naive = explain.assess(evaluation(0.532, 0.516, rows=4000, effective=4000),
                           controls=controls(0.002), walk_forward=folds(6))

    assert naive.trusted, "a large effective sample should clear the bar"
    assert not honest.trusted, (
        "discounting for cross-sectional correlation has to be able to "
        "close the gate, or it is decoration")


def test_a_model_stuck_on_one_answer_is_not_trusted():
    trust = explain.assess(evaluation(0.516, 0.516, up_rate=0.99))
    assert not trust.trusted
    assert "same way almost every day" in trust.reason


def test_an_edge_smaller_than_the_seed_spread_is_not_trusted():
    """The regression test for three runs scoring 51.44, 51.64 and 51.78.

    Sized so the edge clears the chance hurdle comfortably and is stopped only
    by the noise floor -- otherwise this would pass for the wrong reason.
    """
    # 1.2 points over 20,000 independent rows beats chance (0.71) easily.
    trust = explain.assess(evaluation(0.528, 0.516, effective=20000),
                           controls=controls(0.0184),
                           walk_forward=folds(6))
    assert named(trust, "beats_chance")["passed"], "should fail on noise, not chance"
    assert not trust.trusted
    assert "random seeds" in trust.reason
    assert not named(trust, "beats_noise_floor")["passed"]


def test_an_edge_that_appears_in_one_window_of_six_is_not_trusted():
    trust = explain.assess(evaluation(0.55, 0.516),
                           controls=controls(0.003),
                           walk_forward=folds(1))
    assert not trust.trusted
    assert "consecutive test windows" in trust.reason


def test_walk_forward_is_only_a_hurdle_when_it_was_run():
    """No walk-forward is not the same as a failed one, and says so."""
    trust = explain.assess(evaluation(0.56, 0.516), controls=controls(0.004))
    assert trust.trusted
    assert "untested" in named(trust, "consistent_across_time")["detail"]


def test_every_hurdle_is_recorded_even_when_one_fails():
    """A gate that says no without saying which question teaches nobody."""
    trust = explain.assess(evaluation(0.515, 0.516))
    assert [c["name"] for c in trust.checks][:3] == [
        "enough_days", "enough_effective_rows", "model_varies"]
    assert all("detail" in c and c["detail"] for c in trust.checks)


# --- what the gate permits -------------------------------------------------

def test_an_untrusted_model_cannot_produce_a_buy():
    trust = explain.assess(evaluation(0.515, 0.516))
    for probability in (0.01, 0.5, 0.87, 0.99):
        verdict, because = explain.rank(probability, trust)
        assert verdict == explain.UNSURE
        assert "not shown an edge" in because


def test_a_trusted_model_calls_both_ways():
    trust = explain.assess(evaluation(0.56, 0.516), controls=controls(0.004),
                           walk_forward=folds(5))
    assert explain.rank(0.80, trust)[0] == explain.BUY
    assert explain.rank(0.20, trust)[0] == explain.NO_BUY


def test_a_trusted_model_still_declines_near_a_coin_flip():
    trust = explain.assess(evaluation(0.56, 0.516), controls=controls(0.004),
                           walk_forward=folds(5))
    assert explain.rank(0.53, trust)[0] == explain.UNSURE


def test_the_call_threshold_is_demanding():
    assert explain.CALL_THRESHOLD >= 0.55


# --- the reasons -----------------------------------------------------------

class Linear:
    """A model whose arithmetic is known, so attribution can be checked."""

    def __init__(self, weights):
        self.weights = np.asarray(weights, dtype=np.float64)

    def probabilities(self, x):
        score = np.asarray(x, dtype=np.float64) @ self.weights
        up = 1.0 / (1.0 + np.exp(-score))
        return np.stack([1.0 - up, up], axis=1)


@pytest.fixture
def scaler():
    return Scaler(mean=np.zeros(3, dtype=np.float32),
                  std=np.ones(3, dtype=np.float32),
                  feature_names=["a", "b", "c"])


def test_a_feature_the_model_ignores_contributes_nothing(scaler):
    model = Linear([1.0, 0.0, 0.5])
    out = explain.contributions(model, scaler, np.array([1.0, 5.0, 1.0]))
    ignored = next(item for item in out if item["feature"] == "b")
    assert abs(ignored["effect"]) < 1e-9


def test_contributions_are_signed_by_direction(scaler):
    model = Linear([2.0, -2.0, 0.0])
    out = explain.contributions(model, scaler, np.array([1.0, 1.0, 0.0]))
    effects = {item["feature"]: item["effect"] for item in out}
    assert effects["a"] > 0, "a positive weight on a positive value pushes up"
    assert effects["b"] < 0


def test_contributions_are_ordered_by_size(scaler):
    model = Linear([3.0, 1.0, 0.2])
    out = explain.contributions(model, scaler, np.array([1.0, 1.0, 1.0]))
    sizes = [abs(item["effect"]) for item in out]
    assert sizes == sorted(sizes, reverse=True)


def test_a_reason_reads_as_a_sentence(scaler):
    model = Linear([3.0, 0.0, 0.0])
    out = explain.contributions(model, scaler, np.array([2.0, 0.0, 0.0]))
    sentence = explain.in_words(out[0])
    assert sentence.endswith(".")
    assert "a" in sentence


def test_a_feature_at_its_average_is_reported_as_making_no_difference(scaler):
    model = Linear([3.0, 1.0, 1.0])
    out = explain.contributions(model, scaler, np.array([0.0, 0.0, 0.0]))
    assert "almost no difference" in explain.in_words(out[0])


def test_what_it_learnt_ranks_features_by_how_much_they_move_it(scaler):
    model = Linear([4.0, 0.1, 0.0])
    rng = np.random.default_rng(0)
    rows = rng.normal(size=(200, 3))

    summary = explain.what_it_learnt(model, scaler, rows)
    assert summary[0]["feature"] == "a"
    assert summary[-1]["feature"] == "c"
    assert summary[-1]["influence"] == 0.0


# --- the money hurdle ------------------------------------------------------

def test_an_accurate_model_that_loses_money_is_not_trusted():
    """The regression test for a hole the other five hurdles did not cover.

    On the 238-symbol relative panel the logistic control clears every accuracy
    test there is -- past chance, past the seed spread, positive in six of six
    walk-forward windows -- and loses 3.2% a year. Without this check the page
    would have printed "Buy now" over it.
    """
    trust = explain.assess(
        evaluation(0.5111, 0.5003, tstat=-0.93, sharpe=-0.68,
                   rows=107165, effective=25520, days=465, up_rate=0.45),
        controls=controls(0.0007), walk_forward=folds(6))

    assert named(trust, "beats_chance")["passed"]
    assert named(trust, "beats_noise_floor")["passed"]
    assert named(trust, "consistent_across_time")["passed"]
    assert not named(trust, "makes_money")["passed"]

    assert not trust.trusted
    assert "not the same as making money" in trust.reason


def test_a_model_that_is_accurate_and_pays_is_trusted():
    trust = explain.assess(
        evaluation(0.56, 0.516, tstat=3.1, sharpe=1.4),
        controls=controls(0.004), walk_forward=folds(5))

    assert trust.trusted
    assert all(check["passed"] for check in trust.checks)


def test_the_money_hurdle_is_recorded_even_when_it_is_the_only_failure():
    trust = explain.assess(
        evaluation(0.56, 0.516, tstat=0.4),
        controls=controls(0.004), walk_forward=folds(5))

    names = [c["name"] for c in trust.checks]
    assert names[-1] == "makes_money"
    assert sum(1 for c in trust.checks if not c["passed"]) == 1


# --- how often chance clears the chance hurdle ------------------------------
#
# The edge is accuracy minus a baseline measured on the same rows, so it is a
# difference of two noisy proportions, and on an absolute target the baseline
# is shared by every name on a date. The hurdle used to size its bar as one
# proportion over the effective rows, which ignored both. These grade models
# with no skill at all -- sticky random views, like a real model whose features
# move slowly -- through evaluate and the gate, exactly as a run would be.

def skill_free_evaluations(kind, *, trials=150, days=300, names=10, seed=0):
    rng = np.random.default_rng(seed)
    calendar = pd.bdate_range("2020-01-01", periods=days)
    dates = np.repeat(calendar, names)
    symbols = np.tile([f"S{i}" for i in range(names)], days)

    graded = []
    for _ in range(trials):
        shared = (rng.normal(0.0, 0.01, size=(days, 1))
                  if kind == "shared_market" else 0.0)
        returns = shared + rng.normal(0.0, 0.01, size=(days, names))
        if kind == "relative":
            ranks = pd.DataFrame(returns).rank(axis=1, pct=True).to_numpy()
            labels = (ranks > 0.5).astype(int)
            returns = returns - returns.mean(axis=1, keepdims=True)
        else:
            labels = (returns > 0).astype(int)

        view = np.empty((days, names))
        view[0] = rng.random(names)
        for t in range(1, days):
            redraw = rng.random(names) > 0.9
            view[t] = np.where(redraw, rng.random(names), view[t - 1])

        result = evaluate.evaluate(
            view.ravel(), labels.ravel(), returns.ravel(), dates, symbols,
            executable_returns=returns.ravel(), train_up_share=0.5, cost=0.0)
        graded.append(result.to_dict())
    return graded


@pytest.fixture(scope="module")
def skill_free():
    """Simulated once and shared: about 450 evaluations is ten seconds."""
    return {kind: skill_free_evaluations(kind, seed=index)
            for index, kind in enumerate(("independent", "shared_market",
                                          "relative"))}


def chance_hurdle(evaluations, *, measured=True):
    """(share that cleared beats_chance, spread of edge / standard error)."""
    passed, ratios = 0, []
    for scored in evaluations:
        if not measured:
            scored = {k: v for k, v in scored.items()
                      if k != "edge_standard_error"}
        trust = explain.assess(scored)
        check = next((c for c in trust.checks if c["name"] == "beats_chance"),
                     None)
        passed += bool(check and check["passed"])
        ratios.append(trust.edge / max(trust.needed / 2.0, 1e-12))
    return passed / len(evaluations), float(np.std(ratios))


def test_the_chance_hurdle_admits_skill_free_models_at_about_its_nominal_rate(
        skill_free):
    """Two standard errors, one-sided, is a 2.3% false-pass rate by design.

    Both halves are checked. The pass rate says the bar is not too low; the
    spread of edge over standard error says it is not too high either, which
    matters because the obvious repair -- doubling the row variance -- would
    have passed the first check while making the relative target 30% too strict.
    """
    for kind, evaluations in skill_free.items():
        rate, spread = chance_hurdle(evaluations)
        assert rate <= 0.06, (
            f"{kind}: a model with no skill cleared the chance hurdle "
            f"{rate:.1%} of the time")
        assert 0.85 <= spread <= 1.15, (
            f"{kind}: edge / standard error spread {spread:.2f}, not near 1")


def test_the_old_single_proportion_bar_let_them_through(skill_free):
    """The companion, and what a stored run from before the fix still gets.

    Without the measured standard error the gate falls back to the old formula,
    and on the same simulated models it over-admits -- badly where the names
    share a market. If this ever comes back calibrated, the test above has
    stopped exercising the problem it was written for.
    """
    rate, spread = chance_hurdle(skill_free["shared_market"], measured=False)
    assert rate >= 0.10
    assert spread >= 1.6

    _, spread = chance_hurdle(skill_free["independent"], measured=False)
    assert spread >= 1.25


def test_the_measured_standard_error_sets_the_bar_when_present():
    trust = explain.assess({**evaluation(0.53, 0.50), "edge_standard_error": 0.02})
    assert trust.needed == pytest.approx(0.04)
    assert not named(trust, "beats_chance")["passed"]


def test_a_stored_run_without_it_keeps_the_old_formula_and_says_so():
    trust = explain.assess(evaluation(0.53, 0.50, effective=2400))
    assert trust.needed == pytest.approx(2.0 * math.sqrt(0.25 / 2400))
    assert named(trust, "beats_chance")["passed"]
    assert "older" in named(trust, "beats_chance")["detail"]
