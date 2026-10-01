"""The window onto a run.

Everything here is in service of one idea: a signal you cannot interrogate is
worth nothing. So the page shows the working, not the conclusion -- what data
went in, what window it covered, what it scored on rows nobody has seen, what
the controls scored on the same rows, and how far from a coin flip today's
answer actually is.

Numbers are shown together and never apart:

  * accuracy, and the baseline of always guessing the class that dominated
    *training* -- the only baseline anybody had in advance
  * the trained model, and the controls trained on the same rows with the same
    hyperparameters
  * the return the label describes, and the return an order could actually
    reach -- which on this panel disagree completely

The controls are the pair that was missing longest, and they ask the question a
single score cannot: would a logistic regression have done the same? On this
panel it does, which is most of what the page has to report.

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

from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.requests import Request

from .. import attention as attention_mod
from .. import auto as auto_mod
from .. import context as context_mod
from .. import briefing as briefing_mod
from .. import chat as chat_mod
from .. import challenge as challenge_mod
from .. import holding as holding_mod
from .. import promote as promote_mod
from .. import prices as prices_mod
from .. import news as news_mod
from .. import paper as paper_mod
from .. import pipeline
from .. import search as search_mod

logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))

# Load the environment here rather than relying on whoever started the process.
# `run.py` does this before importing uvicorn, but the app can also be launched
# straight through `uvicorn trader.web.app:app` -- from an editor, a launch
# config, or a process manager -- and then it could not. The symptom was silent
# and confusing: every run submitted from the page failed with "No submitter
# key" while the same run from the command line worked, because only one of the
# two entry points had read env/.env.
load_dotenv(os.path.join(HERE, "..", "..", "..", "env", ".env"))
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


class AutoTrain(BaseModel):
    """How often to wake, and where to train when a session has closed.

    The interval is how long the loop may sit after a session closes before it
    notices -- not how often it trains. See auto.py: training again on rows
    that have not changed produces the same model and another look at the
    sealed test period.
    """

    interval_minutes: float = auto_mod.DEFAULT_INTERVAL_MINUTES


def _newest_done():
    done = [r for r in pipeline.list_runs() if r.status == "done"]
    return done[0] if done else None


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
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


# Every signal carries its 46 raw feature values and the model's contribution
# from all 46, which is 1.4 MB of JSON for 238 symbols -- re-sent on every poll
# to draw three sentences per name. `detail` decides how much of that travels:
# "none" for a page that only wants the score, "brief" (the default) for the
# three reasons that are actually rendered, "full" for the raw row.
def _signals(run, detail: str) -> list:
    if detail == "none":
        return []
    if detail == "full":
        return run.signals

    return [{k: v for k, v in signal.items()
             if k not in ("features", "all_contributions")}
            for signal in (run.signals or [])]


@app.get("/api/analysis")
def api_analysis(detail: str = "brief"):
    """The newest finished run, with its reasoning.

    Only finished runs: a model still training has no opinion, and showing the
    previous run's calls beside a "training" badge would invite reading stale
    numbers as current ones.
    """
    if detail not in ("none", "brief", "full"):
        return JSONResponse(
            {"error": "detail must be none, brief or full"}, status_code=400)
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
            # What the model had to beat, measured on the same rows with the
            # same hyperparameters. Shown beside it rather than under it.
            "controls": run.controls,
            "walk_forward": run.walk_forward,
            "verdict": run.verdict,
            "learnt": run.learnt,
            "signals": _signals(run, detail),
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


@app.get("/api/decision")
def api_decision():
    """Everything the front page needs, and nothing it does not.

    The front page polls. Handing it the whole analysis payload meant 1.4 MB
    every fifteen seconds to render one verdict and one decision, so this
    assembles exactly that: the gate, the headline numbers, what the funded
    book intends, and the reasons for the names actually involved.
    """
    run = _newest_done()
    book = None
    try:
        symbols = sorted(holding_mod.symbols_to_price())
        frames = prices_mod.load_many(symbols, period="2y") if symbols else {}
        book = holding_mod.account(frames)
    except Exception as exc:                            # noqa: BLE001
        logger.warning("Could not read the book: %s", exc)

    if run is None:
        return {"run": None, "book": book,
                "why": "No finished model yet. Train one first."}

    # The names the decision is about -- the ones being bought, or held.
    involved = set()
    for entry in ((book or {}).get("waiting_to_buy") or {}).get("buy", []):
        involved.add(entry.get("symbol"))
    for entry in ((book or {}).get("position") or {}).get("bought", []):
        involved.add(entry.get("symbol"))

    reasons = {
        signal["symbol"]: {
            "probability_up": signal.get("probability_up"),
            "confidence": signal.get("confidence"),
            "leaning": signal.get("leaning"),
            "as_of": signal.get("as_of"),
            "close": signal.get("close"),
            "stale_days": signal.get("stale_days"),
            "reasons": signal.get("reasons") or [],
        }
        for signal in (run.signals or []) if signal.get("symbol") in involved
    }

    evaluation = run.evaluation or {}
    return {
        "run": {
            "run_id": run.run_id,
            "created": run.created,
            "horizon": run.horizon,
            "trust": run.trust,
            "verdict": run.verdict,
            "symbols": evaluation.get("symbols"),
            "headline": {
                "accuracy": evaluation.get("accuracy"),
                "baseline_accuracy": evaluation.get("baseline_accuracy"),
                "edge": evaluation.get("edge"),
                "edge_standard_error": evaluation.get("edge_standard_error"),
                "days": evaluation.get("days"),
                "effective_rows": evaluation.get("effective_rows"),
                "executable_tstat": evaluation.get("executable_tstat"),
                "executable_sharpe": evaluation.get("executable_sharpe"),
                "executable_annualised": evaluation.get("executable_annualised"),
                "strategy_sharpe": evaluation.get("strategy_sharpe"),
                "execution_gap": evaluation.get("execution_gap"),
                "test_from": (run.dataset or {}).get("test", {}).get("from"),
                "test_to": (run.dataset or {}).get("test", {}).get("to"),
            },
        },
        "book": book,
        "signals": reasons,
        "briefing": briefing_mod.read(limit=1),
    }


@app.get("/news", response_class=HTMLResponse)
def news_page(request: Request):
    """What the archive is loud about, and what it thinks is missing.

    Its own page because it answers a question the model is structurally
    unable to ask. The news block refuses to become a feature until it has a
    year of history, so for a year the archive is an input to nothing -- this
    is the page that reads it anyway, and says plainly that it is reading.
    """
    return templates.TemplateResponse("news.html",
                                      {"request": request, "v": _static_version()})


@app.get("/api/attention")
def api_attention():
    """The archive's own opinion, with the evidence under every line."""
    run = _newest_done()
    symbols = (run.watchlist if run else None) or pipeline.default_watchlist()

    try:
        # The price join is only interesting for the handful of names that are
        # actually loud, so the shortlist is drawn first and prices fetched for
        # those -- rather than loading a few hundred frames to use eight.
        shortlist = attention_mod.loudest(symbols)
        frames = {}
        if shortlist:
            frames = prices_mod.load_many(
                [row["symbol"] for row in shortlist], period="3mo")
        return attention_mod.opinion(symbols, frames)
    except Exception as exc:                            # noqa: BLE001
        logger.exception("Could not read the archive")
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request):
    """The transcript: what it said unprompted, and what it was asked."""
    return templates.TemplateResponse("chat.html",
                                      {"request": request, "v": _static_version()})


@app.get("/api/chat")
def api_chat(limit: int = 50):
    """The transcript, newest first, plus whether anything can speak."""
    return {"feed": chat_mod.feed(limit=limit),
            "backend": context_mod.backend()}


class Question(BaseModel):
    question: str
    symbol: str | None = None


@app.post("/api/chat")
def api_chat_ask(asked: Question):
    """Ask it something. Recorded either way, answered if a model is running."""
    try:
        return chat_mod.ask(asked.question, symbol=asked.symbol,
                            run=_newest_done())
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:                            # noqa: BLE001
        logger.exception("Could not answer")
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/attention/prose")
def api_attention_prose():
    """One paragraph about the archive, written by whichever model answers.

    The findings are recomputed here rather than accepted from the caller.
    Taking them over the wire would mean the browser chooses what goes into
    the prompt, and a page that can be asked to put arbitrary text in front of
    a model is a page with a prompt-injection hole in it.

    Served separately from /api/attention so the measured page renders
    immediately and the reading arrives when it arrives -- a local model on a
    busy card can take a few seconds, and none of the substance waits on it.
    """
    run = _newest_done()
    symbols = (run.watchlist if run else None) or pipeline.default_watchlist()

    try:
        shortlist = attention_mod.loudest(symbols)
        frames = {}
        if shortlist:
            frames = prices_mod.load_many(
                [row["symbol"] for row in shortlist], period="3mo")
        return context_mod.read_archive(
            attention_mod.opinion(symbols, frames))
    except Exception as exc:                            # noqa: BLE001
        logger.exception("Could not write the archive reading")
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/trials")
def api_trials(limit: int = 40):
    """The search ledger: every configuration asked about, in the order asked.

    Append-only and pre-registered, which is the only reason the number of
    trials can be used to correct the bar the next one has to clear. Served in
    full rather than ranked, because the ones that failed are what make the
    ones that passed mean anything.
    """
    try:
        trials = search_mod.read_trials()
        return {
            "trials": trials[-limit:][::-1],
            "total": len(trials),
            "seal": search_mod.seal_state(),
        }
    except Exception as exc:                            # noqa: BLE001
        logger.exception("Could not read the trial ledger")
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/pnl", response_class=HTMLResponse)
def pnl(request: Request):
    """The forward paper record: what it would have done, and what it cost.

    Its own page because it answers a question neither of the others can. The
    front page asks whether the model scored well on history it was held out
    from; this asks what happened after the prediction was made, which is the
    one measurement in the project that cannot be mined.
    """
    return templates.TemplateResponse("pnl.html",
                                      {"request": request, "v": _static_version()})


@app.get("/api/pnl")
def api_pnl():
    """The equity curve, the statistics, and whether live matches the backtest."""
    state = paper_mod.account()
    run = _newest_done()

    return JSONResponse({
        "account": state,
        "divergence": paper_mod.divergence(state, run.evaluation if run else {}),
        "backtest": {
            "run_id": run.run_id if run else None,
            "executable_sharpe": (run.evaluation or {}).get("executable_sharpe")
            if run else None,
            "executable_annualised": (run.evaluation or {}).get(
                "executable_annualised") if run else None,
            "gate_open": bool((run.trust or {}).get("trusted")) if run else False,
        },
        "cost_per_side": paper_mod.COST_PER_SIDE,
    })


@app.post("/api/pnl/settle")
def api_pnl_settle(background: BackgroundTasks):
    """Fill anything whose session has happened. Safe to call repeatedly."""
    background.add_task(pipeline.settle_paper)
    return {"status": "settling"}


@app.get("/api/book")
def api_book(book_id: str | None = None):
    """A book: what it holds, and when it decides again.

    The funded one by default; `?book_id=` reads a challenger instead.
    """
    try:
        symbols = sorted(holding_mod.symbols_to_price())
        frames = prices_mod.load_many(symbols, period="2y") if symbols else {}
        return holding_mod.account(frames,
                                   book_id=book_id or holding_mod.funded())
    except Exception as exc:                            # noqa: BLE001
        logger.exception("Could not read the held book")
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/books")
def api_books():
    """Every book in the contest, the funded one first."""
    try:
        symbols = sorted(holding_mod.symbols_to_price())
        frames = prices_mod.load_many(symbols, period="2y") if symbols else {}
        return holding_mod.accounts(frames)
    except Exception as exc:                            # noqa: BLE001
        logger.exception("Could not read the contest")
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/briefing")
def api_briefing(limit: int = 14):
    """What it noticed, newest first."""
    return briefing_mod.read(limit=limit)


@app.get("/api/contest")
def api_contest():
    """The books, what the promotion rule makes of them, and the guard."""
    try:
        symbols = sorted(holding_mod.symbols_to_price())
        frames = prices_mod.load_many(symbols, period="2y") if symbols else {}
        return {"books": holding_mod.accounts(frames),
                "funded": holding_mod.funded(),
                "promotion": promote_mod.decide(),
                "guard": promote_mod.drift(),
                "asked": challenge_mod.spent_this_week(),
                "budget": challenge_mod.BUDGET_PER_WEEK,
                "waiting": len(challenge_mod.candidates())}
    except Exception as exc:                            # noqa: BLE001
        logger.exception("Could not read the contest")
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/book/advance")
def api_book_advance(background: BackgroundTasks):
    """Fill what is planned, or sell if the review session has arrived."""
    background.add_task(pipeline.follow_book)
    return {"status": "working"}


@app.get("/api/runs")
def api_runs():
    """Every run, newest first. The page polls this."""
    runs = pipeline.list_runs()
    return JSONResponse([
        {
            "run_id": r.run_id,
            "created": r.created,
            "status": r.status,
            # What it is doing right now. Without it a run that is working and
            # a run that died look identical for minutes at a time.
            "progress": r.progress,
            "silent_for": round(r.silent_for),
            "watchlist": r.watchlist,
            "horizon": r.horizon,
            "spec": r.spec,
            "dataset": r.dataset,
            "evaluation": r.evaluation,
            "controls": r.controls,
            "walk_forward": r.walk_forward,
            "verdict": r.verdict,
            # The gate, so the card can lead with the conclusion rather than
            # with a grid of numbers the reader has to add up themselves.
            "trust": r.trust,
            "signals": r.signals,
            "error": r.error,
            "has_model": r.has_model,
        }
        for r in runs
    ])


@app.get("/api/status")
def api_status():
    runs = pipeline.list_runs()
    return {
        "runs": len(runs),
        "active": sum(1 for r in runs if r.status not in ("done", "failed")),
        "universe": len(pipeline.default_watchlist()),
    }


@app.post("/api/runs")
def api_start(background: BackgroundTasks):
    """Build a dataset and train on it. Returns immediately; the page polls.

    It goes through a background task rather than blocking the request:
    assembling a wide panel takes long enough that a browser would give up.
    """
    if not _starting.acquire(blocking=False):
        return JSONResponse(
            {"error": "A run is already being prepared."}, status_code=409)

    def build_and_train():
        try:
            pipeline.start()
        finally:
            _starting.release()

    background.add_task(build_and_train)
    return {"status": "started"}


@app.get("/api/auto")
def api_auto():
    """What the scheduler is doing, and what it did on its last few cycles."""
    return auto_mod.state()


@app.post("/api/auto/start")
def api_auto_start(body: AutoTrain | None = None):
    settings = body or AutoTrain()
    if settings.interval_minutes < 1:
        return JSONResponse(
            {"error": "an interval below a minute only re-reads the price "
                      "cache faster; a session closes once a day"},
            status_code=400)

    return auto_mod.start(interval_minutes=settings.interval_minutes)


@app.post("/api/auto/stop")
def api_auto_stop():
    """Ask it to finish. The cycle in flight completes; nothing new starts."""
    return auto_mod.stop()


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
