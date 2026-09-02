"""The gate, and the reasons underneath it.

The most dangerous thing this project could do is print "Buy now" over a model
that has learned nothing. Every model it has trained so far is exactly that:
51.2% against a 51.6% baseline, no edge at all. A page that ranked those
probabilities anyway would be a random number generator with confident
typography, and it would be read by the person who built it, who would like it
to work.

So a call is gated on evidence, and the bar is computed rather than chosen: the
accuracy has to beat the baseline by more than two standard errors of a
proportion measured over that many days. Over 400 days that is about five
percentage points; over 4,000 it is about 1.5. These check that the gate holds
in both directions -- that it suppresses when it should, and that it does not
suppress a model that has genuinely earned an opinion.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trader import explain                                    # noqa: E402
from trader.dataset import Scaler                             # noqa: E402


def evaluation(accuracy, baseline, rows=4000, up_rate=0.5):
    return {"accuracy": accuracy, "baseline_accuracy": baseline,
            "rows": rows, "up_rate": up_rate}


# --- the gate --------------------------------------------------------------

def test_no_edge_is_not_trusted():
    trust = explain.assess(evaluation(0.515, 0.516, rows=4187))
    assert not trust.trusted
    assert "no better" in trust.reason


def test_an_edge_smaller_than_chance_is_not_trusted():
    """The real case: a couple of points that look like something and are not."""
    trust = explain.assess(evaluation(0.525, 0.516, rows=4000))

    assert not trust.trusted, (
        "0.9 points over 4,000 days is inside what chance produces")
    assert "chance alone" in trust.reason


def test_a_clear_edge_is_trusted():
    trust = explain.assess(evaluation(0.560, 0.516, rows=4000))
    assert trust.trusted
    assert trust.edge > trust.needed


def test_the_bar_rises_when_there_is_less_evidence():
    """Four points is nothing over 400 days and a lot over 40,000."""
    few = explain.assess(evaluation(0.556, 0.516, rows=400))
    many = explain.assess(evaluation(0.556, 0.516, rows=40000))

    assert few.needed > many.needed, "the bar should move with the sample size"
    assert many.trusted, "the same edge over a hundred times the days should pass"


def test_too_few_days_is_never_trusted_however_good_it_looks():
    trust = explain.assess(evaluation(0.90, 0.50, rows=100))
    assert not trust.trusted
    assert "cannot be told apart from luck" in trust.reason


def test_a_model_stuck_on_one_answer_is_not_trusted():
    """98.8% "up" was the real behaviour, and its accuracy was the class balance."""
    trust = explain.assess(evaluation(0.60, 0.52, rows=4000, up_rate=0.988))

    assert not trust.trusted, (
        "a model that answers the same way every day has an accuracy that is "
        "just the class balance, however far above the baseline it looks")
    assert "learned nothing" in trust.reason


# --- what the gate does to a call -----------------------------------------

def test_an_untrusted_model_cannot_produce_a_buy():
    """Even at total confidence. This is the whole point of the gate."""
    untrusted = explain.assess(evaluation(0.515, 0.516, rows=4187))

    for probability in (0.0, 0.3, 0.5, 0.9, 0.999):
        verdict, because = explain.rank(probability, untrusted)
        assert verdict == explain.UNSURE, (
            f"P(up)={probability} produced {verdict} from a model with no edge")
        assert "Still collecting data" in because


def test_a_trusted_model_calls_both_ways():
    trusted = explain.assess(evaluation(0.560, 0.516, rows=4000))
    assert trusted.trusted

    assert explain.rank(0.85, trusted)[0] == explain.BUY
    assert explain.rank(0.15, trusted)[0] == explain.NO_BUY


def test_a_trusted_model_still_declines_near_a_coin_flip():
    trusted = explain.assess(evaluation(0.560, 0.516, rows=4000))

    verdict, because = explain.rank(0.52, trusted)
    assert verdict == explain.UNSURE
    assert "coin flip" in because, (
        "an unsure call from a good model should say why it is unsure, which is "
        "a different reason from the gate being shut")


def test_the_call_threshold_is_demanding():
    """On a target this noisy, a small lean is the model rounding."""
    assert explain.CALL_THRESHOLD >= 0.55, (
        "calling anything above 0.5 would turn noise into recommendations")


# --- the reasons -----------------------------------------------------------

class Linear:
    """A model whose behaviour is known, so attribution can be checked."""

    def __init__(self, weights):
        self.weights = np.asarray(weights, dtype=np.float32)

    def probabilities(self, x):
        score = np.asarray(x, dtype=np.float32) @ self.weights
        up = 1.0 / (1.0 + np.exp(-score))
        return np.stack([1.0 - up, up], axis=1)


def a_scaler(names):
    return Scaler(mean=np.zeros(len(names), dtype=np.float32),
                  std=np.ones(len(names), dtype=np.float32),
                  feature_names=list(names))


def test_a_feature_the_model_ignores_contributes_nothing():
    model = Linear([2.0, 0.0])
    scaler = a_scaler(["matters", "ignored"])

    found = {c["feature"]: c["effect"]
             for c in explain.contributions(model, scaler, np.array([1.0, 1.0]))}

    assert abs(found["ignored"]) < 1e-6, "a zero weight moved the answer"
    assert found["matters"] > 0.01, "the feature that decides was not credited"


def test_contributions_are_signed_by_direction():
    model = Linear([2.0, -2.0])
    scaler = a_scaler(["pushes_up", "pushes_down"])

    found = {c["feature"]: c["effect"]
             for c in explain.contributions(model, scaler, np.array([1.0, 1.0]))}

    assert found["pushes_up"] > 0
    assert found["pushes_down"] < 0


def test_contributions_are_ordered_by_size():
    model = Linear([0.5, 3.0, 1.0])
    scaler = a_scaler(["small", "large", "middle"])

    order = [c["feature"]
             for c in explain.contributions(model, scaler, np.ones(3))]

    assert order[0] == "large", "the biggest mover should be first"


def test_a_reason_reads_as_a_sentence():
    model = Linear([3.0, 0.0])
    scaler = a_scaler(["return_1d", "quiet"])

    top = explain.contributions(model, scaler, np.array([2.0, 0.0]))[0]
    sentence = explain.in_words(top)

    assert "return 1d" in sentence, "the underscore should not reach a reader"
    assert "towards up" in sentence
    assert sentence.endswith(".")


def test_a_feature_at_its_average_is_reported_as_making_no_difference():
    """Muting a feature already at average changes nothing, and should say so."""
    model = Linear([3.0, 3.0])
    scaler = a_scaler(["moved", "average"])

    found = {c["feature"]: c for c in
             explain.contributions(model, scaler, np.array([1.5, 0.0]))}

    assert "no difference" in explain.in_words(found["average"])


def test_what_it_learnt_ranks_features_by_how_much_they_move_it():
    model = Linear([3.0, 0.0, 1.0])
    scaler = a_scaler(["loud", "unused", "quiet"])

    rows = np.random.default_rng(0).normal(0, 1, (200, 3)).astype(np.float32)
    ranked = explain.what_it_learnt(model, scaler, rows)

    assert [f["feature"] for f in ranked][:1] == ["loud"]
    assert ranked[-1]["feature"] == "unused"
    assert ranked[-1]["influence"] == pytest.approx(0.0, abs=1e-6)
