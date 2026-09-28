"""Training on a schedule, and the two things that makes easy to get wrong.

The first is stopping. A loop you cannot stop from the page that started it is
worse than no loop, and a state file that says "running" after the process died
is a lie the page will repeat for ever.

The second is what a cycle does when there is nothing new. Retraining the same
configuration on the same rows produces the same model up to the seed, and
scores it against the sealed test period again -- so the honest cycle is the one
that notices no session has closed and does the cheap, time-sensitive work
instead. These pin that: training happens once per session, not once per wake.

Nothing here trains a real model or touches the network; pipeline is stubbed.
"""

import datetime as dt
import inspect
import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trader import auto                                       # noqa: E402


class FakeRun:
    def __init__(self, run_id="run-1", status="done"):
        self.run_id = run_id
        self.status = status
        self.local_evaluation = {"edge": 0.01, "executable_tstat": -1.2}
        self.evaluation = None
        self.trust = {"trusted": False, "reason": "test"}


@pytest.fixture
def scheduler(tmp_path, monkeypatch):
    """A state file of its own, a stopped loop, and no real pipeline."""
    monkeypatch.setattr(auto, "STATE_FILE", str(tmp_path / "auto.json"))
    monkeypatch.setattr(auto, "TICK_SECONDS", 0.01)
    monkeypatch.setattr(auto, "HEARTBEAT_SECONDS", 0.02)
    auto._stop.clear()
    monkeypatch.setattr(auto, "_thread", None)
    monkeypatch.setattr(auto, "STALE_SECONDS", 300)

    calls = {"trained": 0, "collected": 0, "settled": 0}

    def train(watchlist=None, **kwargs):
        calls["trained"] += 1
        return FakeRun(f"run-{calls['trained']}")

    def collect_all(**kwargs):
        calls["collected"] += 1
        return []

    def settle(watchlist=None):
        calls["settled"] += 1
        return {"settled": 2, "pending": 1}

    monkeypatch.setattr(auto.pipeline_mod, "start", train)
    monkeypatch.setattr(auto.pipeline_mod, "collect_all", collect_all)
    monkeypatch.setattr(auto.pipeline_mod, "settle_paper", settle)
    monkeypatch.setattr(auto.pipeline_mod, "follow_book",
                        lambda *a, **k: {"note": "stubbed"})
    monkeypatch.setattr(auto, "last_closed_session", lambda: "2026-09-25")

    yield calls

    auto._stop.set()
    thread = auto._thread
    if thread is not None:
        thread.join(timeout=5)
    auto._stop.clear()


# --- what a cycle does --------------------------------------------------------

def test_a_session_that_has_already_been_trained_on_is_not_trained_again(
        scheduler):
    """The whole pacing argument: same rows, same model, another look at the
    sealed test period for nothing."""
    first = auto.cycle(trained_for=None)
    assert first["trained"] == "run-1"
    assert first["trained_for"] == "2026-09-25"

    second = auto.cycle(trained_for="2026-09-25")
    assert second["trained"] is None
    assert "no session has closed" in second["skipped"]
    assert scheduler["trained"] == 1


def test_a_new_session_is_trained_on(scheduler, monkeypatch):
    auto.cycle(trained_for=None)
    monkeypatch.setattr(auto, "last_closed_session", lambda: "2026-09-28")

    out = auto.cycle(trained_for="2026-09-25")
    assert out["trained"] == "run-2"
    assert out["trained_for"] == "2026-09-28"


def test_the_cheap_work_happens_even_on_an_idle_cycle(scheduler):
    """Collecting a finished model and settling yesterday's paper positions are
    time-sensitive in a way training is not."""
    out = auto.cycle(trained_for="2026-09-25")

    assert out["skipped"]
    assert scheduler["collected"] == 1
    assert scheduler["settled"] == 1
    assert out["settled"] == 2 and out["pending"] == 1


def test_a_failed_cycle_is_recorded_rather_than_raised(scheduler, monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("the panel would not build")

    monkeypatch.setattr(auto.pipeline_mod, "start", explode)
    out = auto.cycle(trained_for=None)
    assert out["trained"] is None
    assert "the panel would not build" in out["error"]


def test_forcing_trains_whatever_the_calendar_says(scheduler):
    out = auto.cycle(trained_for="2026-09-25", force=True)
    assert out["trained"] == "run-1"


# --- starting and stopping ----------------------------------------------------

def wait_until(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_it_starts_runs_and_stops(scheduler):
    auto.start(interval_minutes=0.001, backend="local")
    assert auto.running()

    assert wait_until(lambda: scheduler["trained"] >= 1)
    state = auto.stop(wait=5)

    assert state["running"] is False
    assert state["cycles"] >= 1
    assert not auto._thread.is_alive()


def test_stopping_stops_the_training_too(scheduler):
    auto.start(interval_minutes=0.001)
    assert wait_until(lambda: scheduler["trained"] >= 1)

    auto.stop(wait=5)
    settled = scheduler["settled"]
    time.sleep(0.2)
    assert scheduler["settled"] == settled, "a cycle ran after the stop"


def test_starting_twice_does_not_start_two_loops(scheduler):
    auto.start(interval_minutes=0.001)
    again = auto.start(interval_minutes=0.001)

    assert again.get("already_running")
    assert threading_count() == 1
    auto.stop(wait=5)


def threading_count():
    import threading
    return sum(1 for t in threading.enumerate() if t.name == "auto-train")


def test_a_stop_asked_for_on_disk_is_obeyed(scheduler):
    """The page and the command line are different processes; the flag has to
    travel through the file or one can never stop the other."""
    auto.start(interval_minutes=0.001)
    assert wait_until(lambda: scheduler["trained"] >= 1)

    auto._stop.clear()                      # as if this process never heard it
    with open(auto._stop_marker(), "w", encoding="utf-8") as handle:
        handle.write("stop")

    assert wait_until(lambda: not auto.running(), timeout=5)


def test_a_loop_whose_process_died_is_not_reported_as_running(scheduler):
    state = auto._read()
    state.update({"running": True, "heartbeat": (
        dt.datetime.now(dt.timezone.utc)
        - dt.timedelta(seconds=auto.STALE_SECONDS + 60)).isoformat(
            timespec="seconds")})
    auto._write(state)

    current = auto.state()
    assert current["running"] is False
    assert "went away" in current["stopped_because"]


def test_the_history_is_kept_short(scheduler, monkeypatch):
    monkeypatch.setattr(auto, "HISTORY", 3)
    auto.start(interval_minutes=0.001)
    assert wait_until(lambda: auto._read().get("cycles", 0) >= 5)
    auto.stop(wait=5)

    assert len(auto._read()["history"]) == 3


# --- the seam -----------------------------------------------------------------

def test_the_scheduler_cannot_reach_the_search_ledger():
    """A loop that recorded search trials would spend the one budget nobody can
    top up, unattended and overnight. Checked on the imports rather than the
    text, because the module explains in prose why it does not do this."""
    import ast

    tree = ast.parse(inspect.getsource(auto))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported.update(alias.name for alias in node.names)

    assert not {name for name in imported if "search" in name}


# --- the page's controls -------------------------------------------------------

@pytest.fixture
def client(scheduler, monkeypatch):
    from fastapi.testclient import TestClient

    from trader.web import app as web

    monkeypatch.setattr(web.auto_mod, "STATE_FILE", auto.STATE_FILE)
    return TestClient(web.app)


def test_the_page_can_start_and_stop_it(client):
    assert client.get("/api/auto").json()["running"] is False

    started = client.post("/api/auto/start", json={"interval_minutes": 1,
                                                   "backend": "local"})
    assert started.status_code == 200
    assert started.json()["running"] is True

    stopped = client.post("/api/auto/stop")
    assert stopped.status_code == 200
    assert stopped.json()["stop_requested"] is True
    assert wait_until(lambda: not auto.running()), "the loop never stopped"


def test_the_page_refuses_a_pointless_interval(client):
    refused = client.post("/api/auto/start", json={"interval_minutes": 0.1})
    assert refused.status_code == 400
    assert "once a day" in refused.json()["error"]
    assert auto.running() is False


def test_the_page_refuses_an_unknown_backend(client):
    refused = client.post("/api/auto/start", json={"backend": "gpu-farm"})
    assert refused.status_code == 400
    assert auto.running() is False
