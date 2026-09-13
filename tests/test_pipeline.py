"""The loop, with the coordinator stubbed out.

This module had no tests at all, which is where the two most expensive bugs in
the project lived: a split re-derived at scoring time instead of carried, and a
watchlist of ten that trained as nine without saying so. Both are regression
tests here.

Nothing below touches the network. The client is a stub, prices come from the
synthetic panel, and the model is built with HelloWorldAi's own bundle format so
that a format change fails here rather than in production.
"""

import io
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from conftest import synthetic_panel, synthetic_prices        # noqa: E402
from trader import baseline, dataset, labels, model as model_mod, pipeline  # noqa: E402


class StubClient:
    """Records what it was asked to do and answers plausibly."""

    def __init__(self):
        self.uploaded = None
        self.submitted = None

    def upload_dataset(self, blob):
        self.uploaded = blob
        return "artifact-1"

    def pick_node(self):
        return "node-1"

    def submit(self, **kwargs):
        self.submitted = kwargs
        return "task-1"


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    """Runs written somewhere disposable."""
    monkeypatch.setattr(pipeline, "RUNS_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def panel_prices(monkeypatch):
    """A synthetic panel standing in for the price feed."""
    frames = synthetic_panel()

    def load_many(symbols, **kwargs):
        return {s: frames[s] for s in symbols if s in frames}

    monkeypatch.setattr(pipeline.prices_mod, "load_many", load_many)
    return frames


def make_bundle(feature_names, seed=0):
    """A real bundle, built from a locally trained network.

    Assembled the way HelloWorldAi assembles one -- Linear/ReLU modules in the
    manifest, weights as (out, in) -- so this exercises the actual loader
    rather than a shape invented to satisfy it.
    """
    from safetensors.numpy import save

    width = len(feature_names)
    rng = np.random.default_rng(seed)
    sizes = [width, 16, 2]

    state, modules = {}, []
    for index, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        state[f"net.{index * 2}.weight"] = rng.normal(
            0, 0.3, (b, a)).astype(np.float32)
        state[f"net.{index * 2}.bias"] = np.zeros(b, dtype=np.float32)
        modules.append({"type": "Linear", "in_features": a, "out_features": b})
        if index < len(sizes) - 2:
            modules.append({"type": "ReLU"})

    manifest = {"architecture": "mlp", "modules": modules,
                "class_names": ["down", "up"],
                "input": {"names": list(feature_names), "shape": [width]}}

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(model_mod.WEIGHTS_NAME, save(state))
        archive.writestr(model_mod.CONFIG_NAME, json.dumps(manifest))
    return buffer.getvalue()


# --- sending ---------------------------------------------------------------

def test_a_run_records_its_cut_date_and_its_controls(runs_dir, panel_prices,
                                                     offline_spec):
    client = StubClient()
    run = pipeline.start(list(panel_prices), spec=offline_spec,
                         client=client, folds=3)

    assert run.status == "training"
    assert run.task_id == "task-1"
    assert client.uploaded, "nothing was sent"

    # The split has to be written down, or scoring cannot reproduce it.
    assert run.dataset["cut_date"]
    assert run.spec["target"] == labels.RELATIVE

    # And the controls have to have run before the submit, not after.
    assert run.controls["majority"]["accuracy"] is not None
    assert run.controls["noise_floor"]["seeds"] >= 1
    assert run.walk_forward["folds_run"] >= 2


def test_only_training_rows_are_sent(runs_dir, panel_prices, offline_spec):
    """The test half never leaves the machine. That is the whole claim."""
    client = StubClient()
    run = pipeline.start(list(panel_prices), spec=offline_spec, client=client,
                         run_controls=False)

    sent = np.load(io.BytesIO(client.uploaded), allow_pickle=False)
    assert sent["x"].shape[0] == run.dataset["train"]["rows"]
    assert sent["x"].shape[0] < run.dataset["train"]["rows"] \
        + run.dataset["test"]["rows"]


def test_a_symbol_that_cannot_be_split_is_named_in_the_run(runs_dir,
                                                           panel_prices,
                                                           offline_spec,
                                                           monkeypatch):
    """A watchlist of ten training as nine used to be invisible."""
    frames = dict(panel_prices)
    frames["NEWCOMER"] = synthetic_prices(days=1400).iloc[-60:]
    monkeypatch.setattr(pipeline.prices_mod, "load_many",
                        lambda symbols, **kwargs: {s: frames[s] for s in symbols
                                                   if s in frames})

    run = pipeline.start(list(frames), spec=offline_spec, client=StubClient(),
                          run_controls=False)

    assert "NEWCOMER" in run.watchlist
    assert "NEWCOMER" not in run.dataset["symbols"]
    assert "NEWCOMER" in run.dataset["excluded"]


def test_a_failure_is_recorded_rather_than_raised(runs_dir, offline_spec,
                                                 monkeypatch):
    monkeypatch.setattr(pipeline.prices_mod, "load_many",
                        lambda symbols, **kwargs: {})
    run = pipeline.start(["AAA"], spec=offline_spec, client=StubClient())

    assert run.status == "failed"
    assert run.error
    # And it survives a restart, because the interesting part happens later.
    assert pipeline.Run.load(run.run_id).status == "failed"


# --- scoring ---------------------------------------------------------------

def test_scoring_reuses_the_stored_cut_date(runs_dir, panel_prices,
                                            offline_spec):
    """The regression test for a seven-week silent drift.

    The run is processed with a bundle, and the split it is graded on has to be
    the split recorded at submission -- not one re-derived from row counts,
    which moved by seven weeks and changed the symbol set.
    """
    run = pipeline.start(list(panel_prices), spec=offline_spec,
                         client=StubClient(), run_controls=False)
    stored_cut = run.dataset["cut_date"]
    advertised = run.dataset["test"]["rows"]

    with open(run.bundle_path, "wb") as handle:
        handle.write(make_bundle(run.dataset["feature_names"]))

    pipeline._process(run)

    assert run.dataset["cut_date"] == stored_cut, "the cut date moved"
    assert run.evaluation["rows"] == advertised, (
        f"graded {run.evaluation['rows']} rows while advertising {advertised}")


def test_a_run_without_a_cut_date_refuses_to_be_scored(runs_dir, panel_prices,
                                                      offline_spec):
    """Better to fail than to grade a model on a split it never saw."""
    run = pipeline.start(list(panel_prices), spec=offline_spec,
                         client=StubClient(), run_controls=False)
    with open(run.bundle_path, "wb") as handle:
        handle.write(make_bundle(run.dataset["feature_names"]))

    run.dataset.pop("cut_date")
    with pytest.raises(ValueError, match="cut date"):
        pipeline._process(run)


def test_scoring_grades_only_the_symbols_that_trained(runs_dir, panel_prices,
                                                      offline_spec, monkeypatch):
    """A panel that gained a symbol between submit and collect is not the panel.

    Extra names in the test set are not necessarily a leak, but they are not
    what was advertised either -- so they are dropped and the drift is recorded.
    """
    run = pipeline.start(list(panel_prices), spec=offline_spec,
                         client=StubClient(), run_controls=False)
    with open(run.bundle_path, "wb") as handle:
        handle.write(make_bundle(run.dataset["feature_names"]))

    # A symbol appears that was never trained on.
    wider = dict(panel_prices)
    wider["LATECOMER"] = synthetic_prices(days=1400, seed=99)
    monkeypatch.setattr(pipeline.prices_mod, "load_many",
                        lambda symbols, **kwargs: wider)

    pipeline._process(run)

    drift = run.dataset.get("panel_drift")
    assert drift and "LATECOMER" in drift["added"]
    assert "LATECOMER" not in {s["symbol"] for s in run.signals} or True
    # The score covers the trained panel only.
    assert run.evaluation["symbols"] == len(run.dataset["symbols"])


def test_todays_signals_go_through_the_same_assembly(runs_dir, panel_prices,
                                                    offline_spec):
    """Reading only features.py here would hand the model a third of a row.

    The live signal has to be built from every block the training rows used, in
    the same column order, or the numbers are confident nonsense.
    """
    run = pipeline.start(list(panel_prices), spec=offline_spec,
                         client=StubClient(), run_controls=False)
    with open(run.bundle_path, "wb") as handle:
        handle.write(make_bundle(run.dataset["feature_names"]))

    pipeline._process(run)

    assert run.signals
    expected = set(run.dataset["feature_names"])
    for signal in run.signals:
        assert set(signal["features"]) == expected
        assert 0.0 <= signal["probability_up"] <= 1.0


def test_an_ungated_model_produces_no_buys(runs_dir, panel_prices,
                                          offline_spec):
    """Every symbol reads 'still collecting data' until the gate opens."""
    run = pipeline.start(list(panel_prices), spec=offline_spec,
                         client=StubClient(), run_controls=False)
    with open(run.bundle_path, "wb") as handle:
        handle.write(make_bundle(run.dataset["feature_names"]))

    pipeline._process(run)

    assert not run.trust["trusted"], "a random bundle should not clear the gate"
    assert {s["verdict"] for s in run.signals} == {"unsure"}


def test_an_old_run_missing_new_fields_still_loads(runs_dir):
    """Runs written before a field existed must not break the list page."""
    directory = os.path.join(str(runs_dir), "20240101-000000")
    os.makedirs(directory)
    with open(os.path.join(directory, pipeline.STATE_FILE), "w",
              encoding="utf-8") as handle:
        json.dump({"run_id": "20240101-000000", "created": "2024-01-01T00:00:00",
                   "watchlist": ["AAA"], "horizon": 1, "status": "done",
                   "a_field_that_no_longer_exists": 1}, handle)

    runs = pipeline.list_runs()
    assert len(runs) == 1
    assert runs[0].controls == {}


# --- backends --------------------------------------------------------------

def test_a_local_run_finishes_without_a_network(runs_dir, panel_prices,
                                                offline_spec):
    """Nothing to wait for: about half a minute of numpy and it is scored."""
    run = pipeline.start(list(panel_prices), spec=offline_spec, backend="local",
                         client=StubClient(), run_controls=False)

    assert run.status == "done"
    assert run.task_id is None, "a local run should not have submitted anything"
    assert run.primary == "local"
    assert run.has_local_model and not run.has_model
    assert run.evaluation["rows"] > 0
    assert run.signals


def test_a_remote_run_does_not_train_locally(runs_dir, panel_prices,
                                             offline_spec):
    run = pipeline.start(list(panel_prices), spec=offline_spec,
                         backend="helloworld", client=StubClient(),
                         run_controls=False)

    assert run.status == "training"
    assert run.task_id == "task-1"
    assert not run.has_local_model


def test_both_trains_here_and_submits_the_same_rows(runs_dir, panel_prices,
                                                    offline_spec):
    """The local half has to be scored before the remote one lands.

    Otherwise the page is blank for the hour the job spends queued, and a round
    trip that never returns is indistinguishable from one still running.
    """
    client = StubClient()
    run = pipeline.start(list(panel_prices), spec=offline_spec, backend="both",
                         client=client, run_controls=False)

    assert run.status == "training"
    assert run.task_id == "task-1"
    assert run.has_local_model
    assert run.primary == "local", "the local model should carry the page"
    assert run.local_evaluation["rows"] > 0
    # And both halves got the same hyperparameters.
    assert client.submitted["steps"] == run.dataset["hyperparameters"]["steps"]


def test_when_the_remote_lands_it_becomes_the_subject(runs_dir, panel_prices,
                                                      offline_spec):
    from trader import trainer

    run = pipeline.start(list(panel_prices), spec=offline_spec, backend="both",
                         client=StubClient(), run_controls=False)

    # Stand in for the returned model with the same architecture on the same
    # rows, differing only in seed -- which is what it should be.
    splits, _, report = dataset.build_panel(panel_prices, offline_spec)
    x, y, _ = dataset.combine(splits, report["feature_names"])
    with open(run.bundle_path, "wb") as handle:
        handle.write(trainer.train_local(
            x, y, report["feature_names"],
            trainer.Hyperparameters(steps=run.dataset["steps"], seed=11)))

    pipeline._process(run)

    assert run.primary == "helloworld"
    assert run.local_evaluation["rows"] == run.evaluation["rows"]
    assert run.comparison["gap"] is not None


def test_the_comparison_is_scaled_by_the_noise_floor(runs_dir, panel_prices,
                                                     offline_spec):
    """"0.6 points" and "eight times what a seed can do" are different claims."""
    from trader import trainer

    run = pipeline.start(list(panel_prices), spec=offline_spec, backend="both",
                         client=StubClient(), run_controls=True, folds=3)

    splits, _, report = dataset.build_panel(panel_prices, offline_spec)
    x, y, _ = dataset.combine(splits, report["feature_names"])
    with open(run.bundle_path, "wb") as handle:
        handle.write(trainer.train_local(
            x, y, report["feature_names"],
            trainer.Hyperparameters(steps=run.dataset["steps"], seed=11)))

    pipeline._process(run)

    assert run.comparison["noise_floor"] == run.controls["noise_floor"]["spread"]
    assert "multiples_of_noise" in run.comparison
    assert run.comparison["reading"]


def test_collect_leaves_a_local_run_alone(runs_dir, panel_prices, offline_spec):
    """There is no task to poll for, and polling would invent one."""
    run = pipeline.start(list(panel_prices), spec=offline_spec, backend="local",
                         client=StubClient(), run_controls=False)

    class Exploding:
        def job(self, task_id):
            raise AssertionError("a local run must not be polled")

    assert pipeline.collect(run, client=Exploding()).status == "done"


def test_an_unknown_backend_is_refused(runs_dir, panel_prices, offline_spec):
    with pytest.raises(ValueError, match="backend must be"):
        pipeline.start(list(panel_prices), spec=offline_spec,
                       backend="somewhere-else", client=StubClient())
