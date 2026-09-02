"""The window onto a run.

Everything here is in service of one idea: a signal you cannot interrogate is
worth nothing. So the page shows the working, not the conclusion -- what data
went in, what window it covered, what the coordinator measured, what this
project measured on rows nobody has seen, and how far from a coin flip today's
answer actually is.

Two numbers are always shown together and never apart:

  * accuracy, and the baseline of always guessing the commoner direction
  * what HelloWorldAi verified, and what this project measured out of time

The second pair matters because they answer different questions. HelloWorldAi
holds back a random slice and asks "did training work at all" -- a real check,
and the reason a node cannot fake a result. This project holds back the *last*
two years and asks "does it work on days that had not happened yet". A model can
pass the first and fail the second, and that gap is the most informative thing
on the page.

There is no order execution here and no broker credentials anywhere in this
project. It produces opinions about direction; acting on them is a separate
decision made by a person.
"""

from __future__ import annotations

import logging
import os
import threading

from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from .. import pipeline
from ..helloworld import Client

logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
app = FastAPI(title="Trader")

app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(HERE, "templates"))

# One run at a time. Building a panel re-reads ten price histories and the
# submit is not idempotent; two clicks should not become two jobs.
_starting = threading.Lock()


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    reachable, detail = Client().reachable()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "coordinator_ok": reachable,
        "coordinator_detail": detail,
        "watchlist": pipeline.DEFAULT_WATCHLIST,
    })


@app.get("/api/runs")
def api_runs():
    """Every run, newest first. The page polls this."""
    runs = pipeline.list_runs()
    return JSONResponse([
        {
            "run_id": r.run_id,
            "created": r.created,
            "status": r.status,
            "task_id": r.task_id,
            "watchlist": r.watchlist,
            "horizon": r.horizon,
            "dataset": r.dataset,
            "verification": r.verification,
            "evaluation": r.evaluation,
            "verdict": r.verdict,
            "signals": r.signals,
            "error": r.error,
            "has_model": r.has_model,
        }
        for r in runs
    ])


@app.get("/api/status")
def api_status():
    reachable, detail = Client().reachable()
    runs = pipeline.list_runs()
    return {
        "coordinator_ok": reachable,
        "coordinator_detail": detail,
        "runs": len(runs),
        "active": sum(1 for r in runs if r.status not in ("done", "failed")),
    }


@app.post("/api/runs")
def api_start(background: BackgroundTasks):
    """Build a dataset and send it. Returns immediately; the page polls."""
    if not _starting.acquire(blocking=False):
        return JSONResponse(
            {"error": "A run is already being prepared."}, status_code=409)

    def build_and_submit():
        try:
            pipeline.start()
        finally:
            _starting.release()

    background.add_task(build_and_submit)
    return {"status": "started"}


@app.post("/api/collect")
def api_collect():
    """Pick up anything that has finished. The page calls this on its timer.

    Doing it here as well as in the watcher means an open page keeps things
    moving without a second process, and closing the page does not lose
    anything -- both paths call the same idempotent function.
    """
    try:
        collected = pipeline.collect_all()
    except Exception as exc:                            # noqa: BLE001
        logger.exception("Collection failed")
        return JSONResponse({"error": str(exc)}, status_code=500)

    return {"checked": len(collected),
            "done": [r.run_id for r in collected if r.status == "done"]}
