"""The neutral band, and the line between training on clear days and grading on them.

`neutral_band` drops the middle of each date's cross-section: names whose
forward return ranked near the median. On the training side that is the point --
the model is not taught to reproduce noise. On the test side it is look-ahead:
which rows survive depends on where their *realised future* return ranked, which
nobody can know when the position is taken. A banded model was graded only on
names already known to have moved a lot, and a search over the band would have
preferred it for that reason alone.

So these hold one line: the band changes which rows a model learns from, and
nothing else -- not the graded rows, not their returns, not the cut date.
"""

import dataclasses
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from conftest import synthetic_panel                          # noqa: E402
from trader import dataset, search                            # noqa: E402

BAND = 0.4


@pytest.fixture(scope="module")
def panel():
    return synthetic_panel()


def prepared_pair(panel, spec, **overrides):
    plain = dataset.prepare(panel, dataclasses.replace(spec, neutral_band=0.0,
                                                       **overrides))
    banded = dataset.prepare(panel, dataclasses.replace(spec, neutral_band=BAND,
                                                        **overrides))
    return plain, banded


def test_the_band_never_decides_which_rows_are_graded(panel, offline_spec):
    plain, banded = prepared_pair(panel, offline_spec)
    cut = dataset.choose_cut_date(plain.assembled, plain.label_frame)

    plain_splits, _ = dataset.split_at(plain, cut)
    banded_splits, _ = dataset.split_at(banded, cut)
    assert [s.symbol for s in plain_splits] == [s.symbol for s in banded_splits]

    for ordinary, band in zip(plain_splits, banded_splits):
        assert ordinary.test_dates.equals(band.test_dates), (
            f"{band.symbol}: the band removed test rows by where their future "
            f"return ranked")
        np.testing.assert_array_equal(ordinary.y_test, band.y_test)
        np.testing.assert_array_equal(ordinary.forward_returns_test,
                                      band.forward_returns_test)
        np.testing.assert_array_equal(ordinary.executable_returns_test,
                                      band.executable_returns_test)


def test_the_band_still_thins_the_training_rows(panel, offline_spec):
    plain, banded = prepared_pair(panel, offline_spec)
    cut = dataset.choose_cut_date(plain.assembled, plain.label_frame)

    plain_splits, _ = dataset.split_at(plain, cut)
    banded_splits, _ = dataset.split_at(banded, cut)

    kept = sum(len(s.y_train) for s in banded_splits)
    total = sum(len(s.y_train) for s in plain_splits)
    assert kept < total
    # About the band's width comes out; six names rank coarsely, so loosely.
    assert 0.2 < 1.0 - kept / total < 0.6

    for ordinary, band in zip(plain_splits, banded_splits):
        assert set(band.train_dates) <= set(ordinary.train_dates)


def test_the_cut_date_does_not_depend_on_the_band(panel, offline_spec):
    """Otherwise a banded run is graded on a slightly different period, which
    is the same leak at a smaller size."""
    plain, banded = prepared_pair(panel, offline_spec)
    assert (dataset.choose_cut_date(plain.assembled, plain.label_frame)
            == dataset.choose_cut_date(banded.assembled, banded.label_frame))
    assert search.three_way_cut(plain) == search.three_way_cut(banded)


def test_the_purge_still_holds_under_the_band(panel, offline_spec):
    """No training row may carry a label that realises past the cut: the last
    `horizon` sessions before it are dropped whether or not the band would have
    dropped them anyway."""
    horizon = 5
    plain, banded = prepared_pair(panel, offline_spec, horizon=horizon)
    cut = dataset.choose_cut_date(plain.assembled, plain.label_frame)

    for ordinary, band in zip(dataset.split_at(plain, cut)[0],
                              dataset.split_at(banded, cut)[0]):
        assert band.horizon == horizon
        sessions = plain.assembled[band.symbol].index
        before = sessions[sessions < cut]
        purged = set(before[-horizon:])
        assert not purged & set(band.train_dates)
        assert band.train_dates.max() <= ordinary.train_dates.max()


def test_a_validation_score_is_graded_on_the_same_rows_with_or_without_the_band(
        panel, offline_spec):
    plain, banded = prepared_pair(panel, offline_spec)
    cut = search.three_way_cut(plain)

    ordinary = search._score_on(plain, cut, plain.spec, period="validation")
    band = search._score_on(banded, cut, banded.spec, period="validation")
    assert ordinary["rows"] == band["rows"]
    assert ordinary["days"] == band["days"]
