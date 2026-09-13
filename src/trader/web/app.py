"""The window onto a run.

Everything here is in service of one idea: a signal you cannot interrogate is
worth nothing. So the page shows the working, not the conclusion -- what data
went in, what window it covered, what the coordinator measured, what this
project measured on rows nobody has seen, and how far from a coin flip today's
answer actually is.

Numbers are shown together and never apart:

  * accuracy, and the baseline of always guessing the class that dominated
    *training* -- the only baseline anybody had in advance
  * what HelloWorldAi verified, and what this project measured out of time
  * the trained model, and the local controls trained on the same rows with
    the same hyperparameters

The last pair is the one that was missing longest. HelloWorldAi holds back a
random slice and asks "did training work at all" -- a real check, and the reason
a node cannot fake a result. This project holds back the *last* two years and
asks "does it work on days that had not happened yet". And the controls ask the
question neither of those can: would a logistic regression have done the same?

There is no order execution here and no broker credentials anywhere in this
project. It produces opinions about direction; acting on them is a separate
decision made by a person.

The context endpoints are deliberately downstream of all of it. They read the
news store and the finished run and write prose; nothing they return can reach a
feature, a label or a score. That seam is the same one drawn around execution,
and for the same reason.
"""

from __future__ import annotations

import logging
import os
import threading

from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.requests import Request

from .. import context as context_mod
from .. import news as news_mod
from .. import pipeline
from ..helloworld import Client

logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
app = FastAPI(title="Trader")

STATIC_DIR = os.path.join(HERE, "static")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=os.path.join(HERE, "templates"))


def _static_version() -> str:
    """A token that changes when any static file does.

    Without it the browser keeps serving the stylesheet and script it cached,
    which during development means editing the page and being shown the old one
    -- and then wondering why a fix did nothing. Computed per request because
    the whole point is to notice a change made while the server was running.
    """
    newest = 0.0
    try:
        for name in os.listdir(STATIC_DIR):
            newest = max(newest, os.path.getmtime(os.path.join(STATIC_DIR, name)))
    except OSError:
        return "0"
    return str(int(newest))


# One run at a time. Building a panel re-reads hundreds of price histories and
# the submit is not idempotent; two clicks should not become two jobs.
_starting = threading.Lock()


class Question(BaseModel):
    """A question for the reading panel."""

    question: str
    symbol: str | None = None


def _newest_done():
    done = [r for r in pipeline.list_runs() if r.status == "done"]
    return done[0] if done else None


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    reachable, detail = Client().reachable()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "coordinator_ok": reachable,
        "coordinator_detail": detail,
        "watchlist": pipeline.default_watchlist(),
        "v": _static_version(),
    })


@app.get("/analysis", response_class=HTMLResponse)
def analysis(request: Request):
    """The trading view: the call, and the argument for it.

    Separate from the front page because it answers a different question. That
    one asks whether the model is any good; this one asks what it thinks and
    why. The order matters -- the verdicts here are gated on the answer there,
    and a page that showed them without it would be the most misleading thing
    in this project.
    """
    return templates.TemplateResponse("analysis.html",
                                      {"request": request, "v": _static_version()})


@app.get("/api/analysis")
def api_analysis():
    """The newest finished run, with its reasoning.

    Only finished runs: a model still training has no opinion, and showing the
    previous run's calls beside a "training" badge would invite reading stale
    numbers as current ones.
    """
    run = _newest_done()
    if run is None:
        return JSONResponse({"run": None,
                             "why": "No finished model yet. Train one first."})

    return JSONResponse({
        "run": {
            "run_id": run.run_id,
            "created": run.created,
            "horizon": run.horizon,
            "spec": run.spec,
            "trust": run.trust,
            "evaluation": run.evaluation,
            # What a model had to beat, measured locally before anything was
            # sent. Shown beside the trained model rather than under it.
            "controls": run.controls,
            "walk_forward": run.walk_forward,
            "verdict": run.verdict,
            "learnt": run.learnt,
            "signals": run.signals,
            "dataset": {
                "test": (run.dataset or {}).get("test", {}),
                "train": (run.dataset or {}).get("train", {}),
                "cut_date": (run.dataset or {}).get("cut_date"),
                "symbols": (run.dataset or {}).get("symbols", []),
                "excluded": (run.dataset or {}).get("excluded", {}),
                "blocks": (run.dataset or {}).get("blocks", {}),
                "feature_names": (run.dataset or {}).get("feature_names", []),
                "panel_drift": (run.dataset or {}).get("panel_drift"),
            },
        },
        "news": news_mod.readiness(run.watchlist or []),
        "context_available": context_mod.available(),
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
            "spec": r.spec,
            "dataset": r.dataset,
            "verification": r.verification,
            "evaluation": r.evaluation,
            "controls": r.controls,
            "walk_forward": r.walk_forward,
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
        "universe": len(pipeline.default_watchlist()),
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
    """Pick up anything that has finished. The page calls this on its timer."""
    try:
        collected = pipeline.collect_all()
    except Exception as exc:                            # noqa: BLE001
        logger.exception("Collection failed")
        return JSONResponse({"error": str(exc)}, status_code=500)

    return {"checked": len(collected),
            "done": [r.run_id for r in collected if r.status == "done"]}


# --- the reading panel -----------------------------------------------------

@app.get("/api/news")
def api_news():
    """How much point-in-time history the store has, and whether it is usable."""
    return news_mod.readiness(pipeline.default_watchlist())


@app.post("/api/news/collect")
def api_news_collect(background: BackgroundTasks):
    """One append-only pass. Safe to call as often as you like."""
    background.add_task(pipeline.collect_news)
    return {"status": "collecting"}


@app.get("/api/context/{symbol}")
def api_context(symbol: str):
    """Background on one symbol, gated the same way the calls are.

    Returns sources whether or not prose is possible, so the panel degrades to
    a citation list rather than disappearing when no key is configured.
    """
    run = _newest_done()
    signal = None
    if run:
        signal = next((s for s in (run.signals or [])
                       if s.get("symbol") == symbol), None)

    return context_mod.explain_symbol(
        symbol,
        signal=signal,
        trust=(run.trust if run else None),
        evaluation=(run.evaluation if run else None))


@app.post("/api/ask")
def api_ask(body: Question):
    """Answer a question from what is stored and what was measured.

    It cannot be talked into a recommendation: the only opinion in this project
    comes from the gate, which is computed rather than written.
    """
    run = _newest_done()
    signal = None
    if run and body.symbol:
        signal = next((s for s in (run.signals or [])
                       if s.get("symbol") == body.symbol), None)

    return context_mod.answer(
        body.question, body.symbol,
        signal=signal,
        trust=(run.trust if run else None),
        evaluation=(run.evaluation if run else None))
