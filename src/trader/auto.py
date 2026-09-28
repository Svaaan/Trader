"""Training on a schedule, paced by new sessions rather than by the clock.

"Can it train continuously while it runs" has a mechanical answer and a real
one. Mechanically there was no scheduler: one button press, one run. The real
answer is that training the same configuration again does not produce a better
model, and this project has the measurements to say so:

  * it stops itself. Early stopping halts around step 1,000 of a 36,000 budget,
    and training to the ceiling scored 51.1% against 53.7% for stopping -- more
    training made it worse, not better.
  * the same rows and the same configuration give the same model, up to the
    seed. Three identical runs spanned 1.8 points of accuracy, which is wider
    than any edge measured here.
  * **every run scores on the sealed test period.** A loop that retrains every
    ten minutes looks at that period every ten minutes, which is precisely what
    the trial ledger in search.py exists to count. Twenty identical runs are
    one piece of evidence and nineteen extra chances to catch a lucky one.

So the loop wakes on a timer but trains on data. A session closes once a day;
until one does, there is nothing to learn that was not there an hour ago, and
the cycle says so rather than burning a look at the test set to find out. What
it does do every time is the work that genuinely is time-sensitive: collect any
model that finished on HelloWorldAi, and settle the paper positions whose
session has now happened -- the forward record that cannot be mined, and the one
thing here that gets better purely by waiting.

Stopping takes effect between cycles. A local fit is about a minute on the wide
panel and is not interrupted halfway; `stop` is not a kill switch, it is a
promise that nothing new starts.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import threading

from . import pipeline as pipeline_mod
from . import prices as prices_mod

logger = logging.getLogger(__name__)

STATE_FILE = os.environ.get(
    "TRADER_AUTO",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                 "data", "auto.json"),
)

# How long between wake-ups. Half an hour is not a guess about the market: it is
# how long the loop may sit after a session closes before it notices. Shorter
# costs nothing except a price cache read, and buys nothing either.
DEFAULT_INTERVAL_MINUTES = 30

# The clock the panel runs on. One symbol rather than the whole universe: the
# question is only whether another session has closed, and prices.py already
# refuses to re-ask the provider for a series it just fetched.
REFERENCE_SYMBOL = "^GSPC"

# How often the wait checks whether it has been told to stop, and how often the
# loop writes that it is still alive.
TICK_SECONDS = 2.0
HEARTBEAT_SECONDS = 30.0

# A loop whose heartbeat is older than this is not running -- its process was
# killed, or the machine went to sleep with the state file saying "running".
STALE_SECONDS = 5 * 60

# How many cycles to keep in the state file, so the page can show what it has
# been doing without the file growing forever.
HISTORY = 20

_lock = threading.Lock()
# Every read-modify-write of the state file goes through this. Without it the
# loop writing "cycle 3 finished" and a stop writing "please stop" race, and
# whichever reads first silently undoes the other.
_state_lock = threading.RLock()
_thread: threading.Thread | None = None
_stop = threading.Event()


def _path() -> str:
    return os.path.abspath(STATE_FILE)


def _stop_marker() -> str:
    """Where a stop is recorded so that another process cannot lose it.

    A field inside the state file would be clobbered by the next heartbeat the
    loop writes, and the loop is usually in a different process from whoever is
    asking it to stop -- the page stopping what the command line started. A file
    that simply exists cannot be overwritten by an unrelated update.
    """
    return _path() + ".stop"


def _blank() -> dict:
    return {"running": False, "stop_requested": False, "cycles": 0,
            "trained_for": None, "history": [], "started_at": None,
            "heartbeat": None, "interval_minutes": DEFAULT_INTERVAL_MINUTES,
            "backend": None, "pid": None}


def _read() -> dict:
    try:
        with open(_path(), encoding="utf-8") as handle:
            stored = json.load(handle)
    except (OSError, ValueError):
        return _blank()
    return {**_blank(), **(stored if isinstance(stored, dict) else {})}


def _write(state: dict) -> dict:
    path = _path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temporary = f"{path}.{os.getpid()}.tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=1)
        os.replace(temporary, path)
    except OSError as exc:
        logger.debug("Could not write the auto-training state: %s", exc)
    return state


def _update(**changes) -> dict:
    """Read, change, write, without another thread undoing it in between."""
    with _state_lock:
        current = _read()
        current.update(changes)
        return _write(current)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _age(stamp: str | None) -> float:
    if not stamp:
        return float("inf")
    try:
        return (dt.datetime.now(dt.timezone.utc)
                - dt.datetime.fromisoformat(stamp)).total_seconds()
    except (TypeError, ValueError):
        return float("inf")


def state() -> dict:
    """What the loop is doing, from the file, with a dead one reported dead."""
    current = _read()
    if current["running"] and _age(current.get("heartbeat")) > STALE_SECONDS:
        return _update(running=False, stopped_because=(
            "the process running it went away; nothing has been training since "
            f"{current.get('heartbeat')}"))
    return current


# --- one cycle ---------------------------------------------------------------

def last_closed_session() -> str | None:
    """The date of the most recent finished session, or None if unknown."""
    try:
        frame = prices_mod.load(REFERENCE_SYMBOL, period="1y")
    except Exception as exc:                            # noqa: BLE001
        logger.warning("Could not read the calendar from %s: %s",
                       REFERENCE_SYMBOL, exc)
        return None
    return str(frame.index[-1].date()) if len(frame) else None


def cycle(*, backend: str = "local", watchlist=None, trained_for: str | None = None,
          force: bool = False) -> dict:
    """Collect, settle, and train if a session has closed since the last one.

    Returns what happened, including why it did not train when it did not. The
    order matters: collecting and settling are cheap and time-sensitive, and
    both are worth doing even on a cycle that has nothing new to train on.
    """
    out: dict = {"at": _now(), "trained": None, "skipped": None, "error": None}

    try:
        collected = pipeline_mod.collect_all()
        out["collected"] = sum(1 for r in collected if r.status == "done")
    except Exception as exc:                            # noqa: BLE001
        logger.warning("Collection failed this cycle: %s", exc)
        out["collected"], out["error"] = 0, f"collect: {exc}"

    settled = pipeline_mod.settle_paper(watchlist)
    out["settled"] = settled.get("settled", 0)
    out["pending"] = settled.get("pending", 0)

    # The committed book: fill what was planned, sell when the review session
    # arrives. Most days this does nothing, which is the strategy.
    step = pipeline_mod.follow_book()
    out["book"] = {k: v for k, v in step.items() if k != "holding"}
    if step.get("holding"):
        out["holding"] = {
            "symbols": [p["symbol"] for p in step["holding"].get("bought", [])],
            "sessions_left": step["holding"].get("sessions_left"),
            "review_on": step["holding"].get("review_on"),
        }

    session = last_closed_session()
    out["session"] = session

    if not force and session is not None and session == trained_for:
        out["skipped"] = (
            f"no session has closed since {session}; training again on the same "
            f"rows would be the same model and another look at the test set")
        return out

    try:
        run = pipeline_mod.start(watchlist, backend=backend)
    except Exception as exc:                            # noqa: BLE001
        logger.exception("Training failed this cycle")
        out["error"] = f"train: {exc}"
        return out

    out["trained"] = run.run_id
    out["status"] = run.status
    out["trained_for"] = session
    evaluation = run.local_evaluation or run.evaluation or {}
    if evaluation:
        out["edge"] = evaluation.get("edge")
        out["executable_tstat"] = evaluation.get("executable_tstat")
    if run.trust:
        out["trusted"] = bool(run.trust.get("trusted"))
    return out


# --- the loop ----------------------------------------------------------------

def _asked_to_stop() -> bool:
    return _stop.is_set() or os.path.exists(_stop_marker())


def _wait(seconds: float) -> bool:
    """Sleep in slices. False as soon as a stop is asked for, here or on disk."""
    deadline = dt.datetime.now() + dt.timedelta(seconds=seconds)
    beat = 0.0
    while dt.datetime.now() < deadline:
        if _stop.wait(TICK_SECONDS):
            return False
        beat += TICK_SECONDS
        if beat >= HEARTBEAT_SECONDS:
            beat = 0.0
            if _asked_to_stop():
                _stop.set()
                return False
            _update(heartbeat=_now())
    return True


def _loop(*, interval_minutes: float, backend: str, watchlist) -> None:
    while not _asked_to_stop():
        done = cycle(backend=backend, watchlist=watchlist,
                     trained_for=_read().get("trained_for"))

        with _state_lock:
            current = _read()
            changes = {
                "cycles": int(current.get("cycles", 0)) + 1,
                "heartbeat": _now(),
                "last": done,
                "history": ([done] + list(current.get("history", [])))[:HISTORY],
                "next_check": (dt.datetime.now(dt.timezone.utc)
                               + dt.timedelta(minutes=interval_minutes)
                               ).isoformat(timespec="seconds"),
            }
            if done.get("trained_for"):
                changes["trained_for"] = done["trained_for"]
            current = _update(**changes)

        logger.info("Cycle %d: %s", current["cycles"],
                    done.get("skipped") or done.get("error")
                    or f"trained {done.get('trained')} ({done.get('status')})")

        if not _wait(interval_minutes * 60.0):
            break

    _clear_stop()
    final = _update(running=False, stop_requested=False, stopped_at=_now())
    logger.info("Auto-training stopped after %d cycle(s)", final.get("cycles", 0))


def _clear_stop() -> None:
    _stop.clear()
    try:
        os.remove(_stop_marker())
    except OSError:
        pass


def start(*, interval_minutes: float = DEFAULT_INTERVAL_MINUTES,
          backend: str = "local", watchlist=None) -> dict:
    """Begin the loop in a background thread. Idempotent within a process."""
    global _thread

    with _lock:
        if _thread is not None and _thread.is_alive():
            return {**state(), "already_running": True}

        current = state()
        if current["running"]:
            # Another process says it is running and its heartbeat is fresh.
            return {**current, "already_running": True}

        _clear_stop()
        _update(running=True, stop_requested=False, started_at=_now(),
                heartbeat=_now(), cycles=0, interval_minutes=interval_minutes,
                backend=backend, pid=os.getpid(), stopped_at=None,
                stopped_because=None)

        _thread = threading.Thread(
            target=_loop, name="auto-train", daemon=True,
            kwargs={"interval_minutes": interval_minutes, "backend": backend,
                    "watchlist": watchlist})
        _thread.start()

    logger.info("Auto-training started: %s backend, checking every %g minute(s)",
                backend, interval_minutes)
    return state()


def stop(*, wait: float = 0.0) -> dict:
    """Ask the loop to finish. Nothing new starts; the current cycle finishes.

    Sets the flag on disk as well as in this process, so the UI can stop a loop
    the command line started and the other way round.
    """
    _stop.set()
    try:
        os.makedirs(os.path.dirname(_stop_marker()), exist_ok=True)
        with open(_stop_marker(), "w", encoding="utf-8") as handle:
            handle.write(_now())
    except OSError as exc:
        logger.debug("Could not write the stop marker: %s", exc)
    _update(stop_requested=True)

    thread = _thread
    if wait and thread is not None and thread.is_alive():
        thread.join(timeout=wait)

    logger.info("Auto-training asked to stop")
    return state()


def running() -> bool:
    return bool(state()["running"])
