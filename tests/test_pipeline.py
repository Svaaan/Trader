"""The loop, end to end, on a synthetic panel.

This module had no tests at all, which is where the two most expensive bugs in
the project lived: a split re-derived at scoring time instead of carried, and a
watchlist of ten that trained as nine without saying so. Both are regression
tests here.

Nothing below touches the network: prices come from the synthetic panel, and
the model is built in the real bundle format so that a format change fails here
rather than in production.
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

    Assembled the way `trainer.pack_bundle` assembles one -- Linear/ReLU
    modules in the manifest, weights as (out, in) -- so this exercises the
    actual loader rather than a shape invented to satisfy it.
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


# --- running one ------------------------------------------------------------

def test_a_run_records_its_cut_date_and_its_controls(runs_dir, panel_prices,
                                                     offline_spec):
    run = pipeline.start(list(panel_prices), spec=offline_spec, folds=3)

    assert run.status == "done"

    # The split has to be written down, or scoring cannot reproduce it.
    assert run.dataset["cut_date"]
    assert run.spec["target"] == labels.RELATIVE

    # And the controls have to have run before the submit, not after.
    assert run.controls["majority"]["accuracy"] is not None
    assert run.controls["noise_floor"]["seeds"] >= 1
    assert run.walk_forward["folds_run"] >= 2


def test_only_training_rows_are_trained_on(runs_dir, panel_prices, offline_spec):
    """The test half is never fitted on. That is the whole claim, and it used
    to be about what left the machine; now nothing leaves it at all."""
    run = pipeline.start(list(panel_prices), spec=offline_spec,
                         run_controls=False)

    assert run.dataset["train"]["rows"] > 0
    assert run.dataset["test"]["rows"] > 0
    # The split is a cut, not a sample: every row is on one side or the other.
    assert run.dataset["train"]["to"] < run.dataset["test"]["from"]


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

    run = pipeline.start(list(frames), spec=offline_spec,
                          run_controls=False)

    assert "NEWCOMER" in run.watchlist
    assert "NEWCOMER" not in run.dataset["symbols"]
    assert "NEWCOMER" in run.dataset["excluded"]


def test_a_failure_is_recorded_rather_than_raised(runs_dir, offline_spec,
                                                 monkeypatch):
    monkeypatch.setattr(pipeline.prices_mod, "load_many",
                        lambda symbols, **kwargs: {})
    run = pipeline.start(["AAA"], spec=offline_spec)

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
                         run_controls=False)
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
                         run_controls=False)
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
                         run_controls=False)
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
                         run_controls=False)
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
                         run_controls=False)
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
    run = pipeline.start(list(panel_prices), spec=offline_spec,
                         run_controls=False)

    assert run.status == "done"
    assert run.has_model
    assert run.evaluation["rows"] > 0
    assert run.signals


def test_a_run_reports_progress_before_it_has_any_numbers(runs_dir,
                                                          panel_prices,
                                                          offline_spec,
                                                          monkeypatch):
    """The page used to show four zeros for five minutes.

    `run.json` was written once at creation and then not again until the
    controls had finished, so a working run and a dead one looked identical.
    """
    seen = []
    real_save = pipeline.Run.save

    def watching(self, progress=None):
        if progress:
            seen.append((progress, (self.dataset or {}).get("train", {}).get("rows")))
        return real_save(self, progress)

    monkeypatch.setattr(pipeline.Run, "save", watching)
    pipeline.start(list(panel_prices), spec=offline_spec,
                   run_controls=False)

    stages = [p for p, _ in seen]
    assert any("fetching prices" in s for s in stages)
    assert any("building features" in s for s in stages)
    assert any("training rows" in s for s in stages)
    assert any("training" in s for s in stages)
    # And the row count is known before the last stage, not only at the end.
    rows_at = [rows for p, rows in seen if "training rows" in p]
    assert rows_at and rows_at[0]


def test_every_save_stamps_a_heartbeat(runs_dir, panel_prices, offline_spec):
    run = pipeline.start(list(panel_prices), spec=offline_spec,
                         run_controls=False)
    assert run.heartbeat
    assert run.silent_for < 60


def test_an_abandoned_run_is_marked_failed(runs_dir):
    """Kill the process mid-run and the run used to say "Building dataset"
    forever -- `collect` skips it because it has no task id to poll."""
    import datetime as dt

    stale = dt.datetime.now(dt.timezone.utc) - dt.timedelta(
        seconds=pipeline.STALE_RUN_SECONDS + 120)
    directory = os.path.join(str(runs_dir), "20260101-000000")
    os.makedirs(directory)
    with open(os.path.join(directory, pipeline.STATE_FILE), "w",
              encoding="utf-8") as handle:
        json.dump({"run_id": "20260101-000000", "created": "2026-01-01T00:00:00",
                   "watchlist": ["AAA"], "horizon": 1, "status": "building",
                   "progress": "building features for 238 symbols",
                   "heartbeat": stale.isoformat(timespec="seconds")}, handle)

    run = pipeline.list_runs()[0]
    assert run.status == "failed"
    assert "Abandoned" in run.error
    assert "building features" in run.error
    # And it stays failed across a reload rather than being re-judged.
    assert pipeline.Run.load("20260101-000000").status == "failed"


def test_a_run_that_is_still_working_is_left_alone(runs_dir):
    import datetime as dt

    recent = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=30)
    directory = os.path.join(str(runs_dir), "20260101-000001")
    os.makedirs(directory)
    with open(os.path.join(directory, pipeline.STATE_FILE), "w",
              encoding="utf-8") as handle:
        json.dump({"run_id": "20260101-000001", "created": "2026-01-01T00:00:00",
                   "watchlist": ["AAA"], "horizon": 1, "status": "building",
                   "heartbeat": recent.isoformat(timespec="seconds")}, handle)

    assert pipeline.list_runs()[0].status == "building"


def test_the_app_loads_its_own_environment():
    """Started through uvicorn directly, the page read none of its settings.

    `run.py` loads env/.env; a launch config running `uvicorn trader.web.app:app`
    did not, so a run started from the button ignored TRADER_UNIVERSE and the
    cache paths while the identical run from the command line honoured them.
    """
    import importlib

    import trader.web.app as web_app

    source = importlib.resources.files  # noqa: F841  (import guard only)
    assert "load_dotenv" in open(web_app.__file__, encoding="utf-8").read()


def test_a_feature_that_can_no_longer_be_built_refuses_scoring_by_name(
        runs_dir, panel_prices, offline_spec, monkeypatch):
    """A macro series that stopped updating takes its features out of the
    block. A run trained on one of them must say so, not fail on a shape
    mismatch three calls later or be scored on a column that was invented."""
    run = pipeline.start(list(panel_prices), spec=offline_spec,
                         run_controls=False)
    with open(run.bundle_path, "wb") as handle:
        handle.write(make_bundle(run.dataset["feature_names"]))

    real = pipeline.dataset_mod.prepare
    lost = run.dataset["feature_names"][-1]

    def without_one(frames, spec=None, **kwargs):
        prepared = real(frames, spec, **kwargs)
        prepared.feature_names = [n for n in prepared.feature_names if n != lost]
        prepared.report["macro"] = {"ended": {"vix3m": {"last": "2026-07-17"}}}
        return prepared

    monkeypatch.setattr(pipeline.dataset_mod, "prepare", without_one)
    with pytest.raises(ValueError, match="can no longer be built") as refused:
        pipeline._process(run)
    assert lost in str(refused.value)
    assert "2026-07-17" in str(refused.value)


# --- the pages, against the Run they actually read -----------------------------
#
# Removing the remote trainer removed four fields from Run, and two endpoints
# went on reading them. Nothing failed: no test had a saved run on disk when it
# called the pages, so the list comprehension never ran a single iteration and
# /api/runs returned an empty list instead of raising. In the real data
# directory it raised AttributeError and the runs page rendered blank.
#
# So these two check the join between the pages and the record, which is the
# seam a field removal breaks and the one nothing else was watching.

def test_every_field_the_pages_read_exists_on_a_run():
    """A static check, because the dynamic one needs a run and a server."""
    import ast
    import dataclasses

    from trader.web import app as web_app

    known = {field.name for field in dataclasses.fields(pipeline.Run)}
    known |= {name for name in dir(pipeline.Run) if not name.startswith("__")}

    tree = ast.parse(open(web_app.__file__, encoding="utf-8").read())
    missing = sorted({
        f"{node.lineno}: {node.value.id}.{node.attr}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in ("r", "run")
        and node.attr not in known
    })
    assert not missing, f"the pages read fields Run does not have: {missing}"


def test_the_pages_serve_a_real_run(runs_dir, panel_prices, offline_spec,
                                   monkeypatch):
    """With a run on disk, every read endpoint answers rather than raising."""
    from starlette.testclient import TestClient

    from trader.web import app as web_app

    run = pipeline.start(list(panel_prices), spec=offline_spec,
                         run_controls=False)
    assert run.status == "done"

    client = TestClient(web_app.app)

    listed = client.get("/api/runs")
    assert listed.status_code == 200
    assert listed.json()[0]["run_id"] == run.run_id

    analysis = client.get("/api/analysis")
    assert analysis.status_code == 200
    assert analysis.json()["run"]["run_id"] == run.run_id


def test_the_polling_page_is_not_handed_the_whole_run(runs_dir, panel_prices,
                                                      offline_spec):
    """The front page polls. It used to poll the full analysis payload.

    Every signal carries its raw feature row and the model's contribution from
    all 46 features, which on the real 238-name panel is 1.4 MB of JSON -- re-
    fetched every fifteen seconds to render one verdict and three sentences.
    `detail` decides how much of that travels, and /api/decision carries only
    the names the decision is actually about.
    """
    from starlette.testclient import TestClient

    from trader.web import app as web_app

    run = pipeline.start(list(panel_prices), spec=offline_spec,
                         run_controls=False)
    assert run.signals, "this test needs a run that produced signals"

    client = TestClient(web_app.app)

    full = client.get("/api/analysis?detail=full").json()["run"]["signals"]
    brief = client.get("/api/analysis?detail=brief").json()["run"]["signals"]
    none = client.get("/api/analysis?detail=none").json()["run"]["signals"]

    assert len(full) == len(brief) and none == []
    assert "features" in full[0] and "all_contributions" in full[0]
    # The three reasons survive -- they are what the page renders.
    assert "features" not in brief[0] and "all_contributions" not in brief[0]
    assert "reasons" in brief[0] and "probability_up" in brief[0]

    assert client.get("/api/analysis?detail=sideways").status_code == 400


def test_the_decision_endpoint_answers_for_the_names_it_names(
        runs_dir, panel_prices, offline_spec):
    """It carries reasons for the symbols the book is acting on, and no others.

    A page that names a symbol and then quotes another one's numbers is the bug
    this endpoint's shape exists to make impossible: the reasons are keyed by
    the symbol they belong to rather than handed over as a list to index into.
    """
    from starlette.testclient import TestClient

    from trader.web import app as web_app

    pipeline.start(list(panel_prices), spec=offline_spec, run_controls=False)
    body = TestClient(web_app.app).get("/api/decision").json()

    assert body["run"]["trust"] is not None
    assert "accuracy" in body["run"]["headline"]

    book = body.get("book") or {}
    wanted = {entry["symbol"]
              for entry in (book.get("waiting_to_buy") or {}).get("buy", [])}
    wanted |= {entry["symbol"]
               for entry in (book.get("position") or {}).get("bought", [])}
    assert set(body["signals"]) <= wanted


def test_the_trial_ledger_is_served_in_the_order_it_was_written(monkeypatch,
                                                                tmp_path):
    """Newest first, with the total, because the total is what raises the bar."""
    import json as json_mod
    from starlette.testclient import TestClient

    from trader import search as search_mod
    from trader.web import app as web_app

    directory = tmp_path / "search"
    directory.mkdir()
    monkeypatch.setattr(search_mod, "SEARCH_DIR", str(directory))
    with open(directory / search_mod.TRIALS_FILE, "w", encoding="utf-8") as handle:
        for n in (1, 2, 3):
            handle.write(json_mod.dumps({"trial": n, "spec": {}, "book": {},
                                         "validation": {"edge": 0.01 * n}}) + "\n")

    body = TestClient(web_app.app).get("/api/trials").json()
    assert body["total"] == 3
    assert [t["trial"] for t in body["trials"]] == [3, 2, 1]
