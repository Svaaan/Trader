"""The whole loop: build a dataset, train on it, score it, and say something.

One pass, one machine, synchronous: fetch prices, build features, cut the panel
once, train here, score on rows the model never saw, run the controls, and hand
the result to the gate. A run finishes before `start` returns.

It used to do this twice -- once here and once on a rented GPU through
HelloWorldAi -- so the two could be compared. That comparison is over: the
network is 7,233 parameters and a logistic regression matches it, so the round
trip was paying for a difference that was not there. What is left is the half
that was doing the work.

**The test half never leaves this machine**, and never did. The split is made
once, carried, and reused for scoring -- see dataset.py for what re-deriving it
costs.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from . import baseline as baseline_mod
from . import dataset as dataset_mod
from . import evaluate as evaluate_mod
from . import explain as explain_mod
from . import features as features_mod
from . import labels as labels_mod
from . import model as model_mod
from . import names as names_mod
from . import news as news_mod
from . import briefing as briefing_mod
from . import attention as attention_mod
from . import chat as chat_mod
from . import challenge as challenge_mod
from . import holding as holding_mod
from . import promote as promote_mod
from . import paper as paper_mod
from . import prices as prices_mod
from . import trainer as trainer_mod
from . import universe as universe_mod

logger = logging.getLogger(__name__)

RUNS_DIR = os.environ.get(
    "TRADER_RUNS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "runs"),
)

# Ten names is where this started and is too narrow for the macro and
# cross-sectional blocks to say anything -- see universe.py. Override with
# TRADER_UNIVERSE=core while iterating on code.
DEFAULT_UNIVERSE = os.environ.get("TRADER_UNIVERSE", "wide")

STATE_FILE = "run.json"
BUNDLE_FILE = "model.zip"
LOCAL_BUNDLE_FILE = "local_model.zip"

# Training length, in passes over the data rather than in steps.
#
# `steps=4000` was the default when the panel was ten symbols and 18,000 rows,
# where it is fourteen passes -- a reasonable amount of training. On 238 symbols
# and 428,000 rows the same number is *six tenths of one pass*, and the local
# control demonstrated exactly what that produces: up-rate 1.00, accuracy equal
# to the class balance, a model that never got far enough from its
# initialisation to learn anything. It would have cost a GPU round trip to find
# that out remotely.
#
# So the step count is derived from the data instead of fixed, and the run
# records what it worked out to.
TARGET_EPOCHS = 12
BATCH_SIZE = 64
MIN_STEPS = 4000


def steps_for(rows: int, *, epochs: int = TARGET_EPOCHS,
              batch: int = BATCH_SIZE) -> int:
    """How many gradient steps `rows` rows are *allowed*.

    A ceiling, not an instruction. `baseline.fit_mlp` holds back the most
    recent slice of the training rows and stops when the validation loss stops
    improving, which on the wide panel happens around 5,000 steps whatever this
    returns. Training to the ceiling instead costs accuracy: measured, 5,000
    steps scores 0.5118 out of time and 200,000 scores 0.5029, while training
    accuracy climbs from 0.5194 to 0.5690 the whole way. That is memorisation,
    and it is why "train for longer" and "loop until it is smarter" have the
    same answer.

    Floored, because a very small panel still needs enough steps to converge,
    and capped so that a very wide one cannot run away. Early stopping makes
    the ceiling cheap: on a 240-symbol panel this allows roughly twenty times
    the old fixed 4,000 and the fit uses a fraction of it. Pass `steps=`
    explicitly to override.
    """
    return int(min(max(epochs * max(rows, 1) // batch, MIN_STEPS), 200_000))


def default_watchlist() -> list:
    return universe_mod.resolve(DEFAULT_UNIVERSE)


# Kept as a name because the UI and older runs refer to it.
DEFAULT_WATCHLIST = universe_mod.CORE


def _runs_root() -> str:
    return os.path.abspath(RUNS_DIR)


def _run_dir(run_id: str) -> str:
    return os.path.join(_runs_root(), run_id)


@dataclasses.dataclass
class Run:
    """One submitted job and everything known about it."""

    run_id: str
    created: str
    watchlist: list
    horizon: int
    status: str = "building"
    spec: dict = dataclasses.field(default_factory=dict)
    dataset: dict = dataclasses.field(default_factory=dict)
    evaluation: dict = dataclasses.field(default_factory=dict)
    controls: dict = dataclasses.field(default_factory=dict)
    walk_forward: dict = dataclasses.field(default_factory=dict)
    # What it is doing right now, and when it last said so. A run used to save
    # once at creation and then not again until the controls had finished, so
    # for five minutes the page showed "Building dataset" over four zeros --
    # indistinguishable from a process that had died. Both fields are written
    # at every stage; `heartbeat` is also what tells a later reader that a run
    # is genuinely working rather than abandoned.
    progress: str = ""
    heartbeat: str = ""
    verdict: str = ""
    signals: list = dataclasses.field(default_factory=list)
    trust: dict = dataclasses.field(default_factory=dict)
    learnt: list = dataclasses.field(default_factory=list)
    error: str = ""

    def save(self, progress: str | None = None) -> None:
        """Write the run to disk, stamping what it is doing and when.

        Called at every stage rather than only at the end. The cost is a few
        kilobytes; the benefit is that the page can show what is happening and
        that an abandoned run can be told from a working one.
        """
        if progress is not None:
            self.progress = progress
        self.heartbeat = dt.datetime.now(dt.timezone.utc).isoformat(
            timespec="seconds")

        directory = _run_dir(self.run_id)
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, STATE_FILE), "w", encoding="utf-8") as fh:
            json.dump(dataclasses.asdict(self), fh, indent=2)

    @property
    def silent_for(self) -> float:
        """Seconds since this run last said anything, or 0 if it never did."""
        if not self.heartbeat:
            return 0.0
        try:
            last = dt.datetime.fromisoformat(self.heartbeat)
        except ValueError:
            return 0.0
        if last.tzinfo is None:
            last = last.replace(tzinfo=dt.timezone.utc)
        return (dt.datetime.now(dt.timezone.utc) - last).total_seconds()

    @classmethod
    def load(cls, run_id: str) -> "Run":
        with open(os.path.join(_run_dir(run_id), STATE_FILE), encoding="utf-8") as fh:
            raw = json.load(fh)
        # Runs written by an earlier version are missing fields added since.
        # Dropping unknown keys and defaulting absent ones means old runs stay
        # readable instead of making the whole list page fail to render.
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    @property
    def bundle_path(self) -> str:
        path = os.path.join(_run_dir(self.run_id), BUNDLE_FILE)
        # Runs from when there were two trainers kept theirs under the other
        # name; they still load.
        if not os.path.exists(path) and os.path.exists(self.local_bundle_path):
            return self.local_bundle_path
        return path

    @property
    def local_bundle_path(self) -> str:
        """Where runs trained before the remote half was removed kept theirs."""
        return os.path.join(_run_dir(self.run_id), LOCAL_BUNDLE_FILE)

    @property
    def has_model(self) -> bool:
        return os.path.exists(self.bundle_path) or os.path.exists(
            self.local_bundle_path)

    @property
    def has_local_model(self) -> bool:
        return os.path.exists(self.local_bundle_path)

    @property
    def cut_date(self) -> Optional[pd.Timestamp]:
        """The split this run was built on. Scoring must reuse it."""
        raw = (self.dataset or {}).get("cut_date")
        return pd.Timestamp(raw) if raw else None


# How long a run may go without saying anything before it is presumed dead.
# Every stage of `start` stamps a heartbeat, and the longest single stage -- the
# walk-forward on a wide panel -- is about four minutes, so fifteen is generous
# enough never to condemn a working run and short enough to clear the list.
STALE_RUN_SECONDS = 15 * 60


def reconcile(runs: list) -> list:
    """Mark abandoned runs as failed instead of leaving them mid-sentence.

    A run whose process was killed -- the server restarted, the machine slept,
    a Ctrl-C during the panel build -- keeps whatever status it had written
    last, and `collect` will not touch it because it has no task id to poll. So
    it sits in the list saying "Building dataset" forever, indistinguishable
    from one that is genuinely working. Measured: restarting the preview server
    mid-run left exactly that, permanently.

    """
    for run in runs:
        if run.status in ("done", "failed"):
            continue
        if run.silent_for <= STALE_RUN_SECONDS:
            continue

        run.status = "failed"
        run.error = (
            f"Abandoned. Nothing was written for "
            f"{run.silent_for / 60:.0f} minutes while it was "
            f"{run.progress or 'starting'}, so the process that was running it "
            f"is gone. Start another.")
        logger.warning("Run %s looks abandoned; marking it failed", run.run_id)
        run.save()

    return runs


def list_runs() -> list:
    """Newest first. A directory that will not parse is skipped, not fatal."""
    root = _runs_root()
    if not os.path.isdir(root):
        return []

    runs = []
    for name in sorted(os.listdir(root), reverse=True):
        try:
            runs.append(Run.load(name))
        except Exception as exc:                        # noqa: BLE001
            logger.warning("Ignoring unreadable run %s: %s", name, exc)

    return reconcile(runs)


# --- running one -----------------------------------------------------------

def start(watchlist: Sequence[str] | None = None, *, horizon: int = 1,
          period: str = "10y", test_fraction: float = 0.2,
          target: str = labels_mod.RELATIVE,
          spec: dataset_mod.Spec | None = None,
          steps: int | None = None,
          run_controls: bool = True, folds: int = 6) -> Run:
    """Build a dataset from live prices, train on it, and score what comes out.

    Synchronous: the run is finished when this returns. Training a 7,233
    parameter network on a few hundred thousand rows is about a minute of
    numpy, which is why the remote trainer this used to have was not worth its
    round trip.

    Pass `spec` to choose which feature blocks go in -- switching one off and
    re-running is how its contribution gets measured rather than assumed. The
    keyword arguments are the common case and are ignored when `spec` is given.

    `steps` defaults to whatever the dataset size deserves -- see `steps_for`.
    A fixed step count is a fixed number of *samples*, which is a shrinking
    number of passes as the panel widens, and an undertrained model looks
    exactly like a model with nothing to learn.
    """
    symbols = universe_mod.resolve(watchlist) if watchlist else default_watchlist()

    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    spec = spec or dataset_mod.Spec(target=target, horizon=horizon,
                                    test_fraction=test_fraction)

    run = Run(run_id=run_id,
              created=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              watchlist=symbols, horizon=horizon, spec=spec.to_dict())
    run.save(f"queued: {len(symbols)} symbols")

    try:
        run.save(f"fetching prices for {len(symbols)} symbols")
        frames = prices_mod.load_many(symbols, period=period)
        if not frames:
            raise ValueError("no price history could be fetched for any symbol")

        run.save(f"building features for {len(frames)} symbols")
        splits, cut, report = dataset_mod.build_panel(frames, spec)
        x_train, y_train, scaler = dataset_mod.combine(
            splits, report["feature_names"])

        description = dataset_mod.describe(splits, scaler, spec, report, cut)
        run.dataset = description
        # The row counts exist now, so the page can stop showing zeros.
        run.save(f"{len(splits)} symbols, {len(y_train):,} training rows")

        # Now that the rows exist, work out how much training they deserve.
        training_steps = steps if steps is not None else steps_for(len(y_train))
        description["steps"] = training_steps
        description["epochs"] = round(
            training_steps * BATCH_SIZE / max(len(y_train), 1), 2)

        # --- what a model has to beat, measured here, before anything is sent --
        #
        # Same width, depth, batch and step count as the job about to be
        # submitted, so the control is a control. This is also the cheapest
        # place to discover that the step count is wrong for the panel size.
        if run_controls:
            try:
                run.save("running the controls (majority, logistic, MLP)")
                run.controls = baseline_mod.run_controls(
                    splits, report["feature_names"], hidden=64, depth=2,
                    steps=training_steps, batch=BATCH_SIZE)
                run.save(f"walking forward over {folds} windows")
                run.walk_forward = baseline_mod.walk_forward(
                    frames, spec, folds=folds)
                run.save("controls done")
            except Exception as exc:                    # noqa: BLE001
                logger.warning("Controls failed, continuing: %s", exc)
                run.controls = {"error": str(exc)}

        hyper = trainer_mod.Hyperparameters(
            hidden=64, depth=2, steps=training_steps, batch=BATCH_SIZE)
        description["hyperparameters"] = hyper.to_dict()

        # --- train ---------------------------------------------------------
        run.save(f"training: {training_steps:,} steps at most")
        bundle = trainer_mod.train_local(
            x_train, y_train, report["feature_names"], hyper)
        with open(run.bundle_path, "wb") as handle:
            handle.write(bundle)
        logger.info("Run %s: trained (%d bytes)", run_id, len(bundle))

        run.save("scoring out of time")
        _process(run)
        run.status = "done"
        run.progress = "finished"
        follow_book(run)
        logger.info("Run %s finished: %s", run_id, run.verdict)

    except Exception as exc:                            # noqa: BLE001
        run.status = "failed"
        run.error = str(exc)
        logger.exception("Run %s could not be started", run_id)

    run.save()
    return run


# --- collecting ------------------------------------------------------------

def collect_all(**_ignored) -> list:
    """Kept because the page still calls it on its timer.

    There is nothing to collect: training happens here and finishes before
    `start` returns. What this still does is notice a run whose process died
    mid-build, so the page shows "abandoned" rather than "building" for ever.
    """
    return reconcile(list_runs())


def _process(run: Run) -> None:
    """Score the run's model out of time, on one rebuild of its own split."""
    scaler = dataset_mod.Scaler.from_dict(run.dataset["scaler"])
    spec = dataset_mod.Spec.from_dict(run.spec or {})

    cut = run.cut_date
    if cut is None:
        raise ValueError(
            "this run recorded no cut date, so its test set cannot be "
            "reproduced. Re-deriving one would grade the model on a different "
            "split than it was trained for.")

    frames = prices_mod.load_many(run.watchlist, period="10y")

    # The split this run was built on, not a fresh one. See the module docstring.
    prepared = dataset_mod.prepare(frames, spec)

    # A feature the model was trained on that the panel can no longer build --
    # a macro series the provider stopped publishing, most likely. Scoring
    # anyway would either fail on a shape mismatch or, worse, need the column
    # invented; say which it is instead.
    unavailable = [name for name in scaler.feature_names
                   if name not in prepared.feature_names]
    if unavailable:
        ended = ((prepared.report.get("macro") or {}).get("ended") or {})
        raise ValueError(
            f"this run was trained on {unavailable}, which can no longer be "
            f"built" + (f" (macro series that stopped updating: "
                        f"{ {k: v['last'] for k, v in ended.items()} })"
                        if ended else "") +
            ". Its test set cannot be scored as trained; retrain on the "
            "features that exist.")

    splits, report = dataset_mod.split_at(prepared, cut)

    # If the panel is not the one that trained, the score is not the one that
    # was advertised. Say so rather than quietly grading something else.
    trained_on = set(run.dataset.get("symbols") or [])
    scoring = {s.symbol for s in splits}
    if trained_on and scoring != trained_on:
        logger.warning(
            "Scoring panel differs from the trained panel: added %s, lost %s",
            sorted(scoring - trained_on), sorted(trained_on - scoring))
        run.dataset["panel_drift"] = {
            "added": sorted(scoring - trained_on),
            "lost": sorted(trained_on - scoring),
        }
        # Grade only what was trained on, so the number means what it says.
        splits = [s for s in splits if s.symbol in trained_on]
        if not splits:
            raise ValueError("none of the trained symbols could be rebuilt")

    # --- the honest score: rows that were never sent anywhere ---
    test = dataset_mod.test_matrix(splits, scaler)
    x_test = test.x
    train_up_share = (run.dataset.get("train") or {}).get("up_share")

    if not run.has_model:
        raise ValueError("this run has no model to score")

    model = model_mod.load_bundle_file(run.bundle_path)
    probabilities = model.probabilities(test.x)[:, 1]
    result = evaluate_mod.evaluate(
        probabilities, test.y, test.returns, test.dates, test.symbols,
        executable_returns=test.executable,
        train_up_share=train_up_share, horizon=test.horizon)

    run.evaluation = result.to_dict()
    run.verdict = evaluate_mod.verdict(result)

    # Whether anything below is worth printing. Decided once, from the
    # out-of-time score, the noise floor and the walk-forward.
    trust = explain_mod.assess(run.evaluation, controls=run.controls,
                               walk_forward=run.walk_forward)
    run.trust = trust.to_dict()

    # What the model attends to in general, which is how a network that has
    # collapsed to one answer shows itself from the inside.
    run.learnt = explain_mod.what_it_learnt(model, scaler, x_test)

    # --- and what it says about today ---
    run.signals = _todays_signals(model, scaler, frames, spec, trust)


def _todays_signals(model, scaler, frames: dict, spec, trust=None) -> list:
    """The most recent finished session, per symbol.

    This is the row with features and no label -- the one prediction exists for.
    Built through the same assembly the training rows went through, so the
    columns are in the same order and mean the same things. Reading only
    features.py here would silently drop the macro, cross-sectional and event
    blocks and hand the model a third of a row.
    """
    signals = []

    try:
        assembled, names, _ = dataset_mod.assemble(frames, spec)
    except Exception as exc:                            # noqa: BLE001
        logger.warning("Could not assemble today's rows: %s", exc)
        return signals

    expected = list(scaler.feature_names)

    for symbol in sorted(assembled):
        try:
            frame = assembled[symbol]
            if frame.empty:
                continue

            missing = [n for n in expected if n not in frame.columns]
            if missing:
                logger.warning("%s is missing %s today; no signal", symbol, missing)
                continue

            latest = frame.iloc[[-1]]
            row = latest[expected].to_numpy(dtype=np.float32)
            probability = float(model.probabilities(scaler.apply(row))[0, 1])

            verdict, because = (explain_mod.rank(probability, trust)
                                if trust is not None else (explain_mod.UNSURE, ""))

            reasons = explain_mod.contributions(model, scaler, row)
            prices = frames.get(symbol)

            signals.append({
                "verdict": verdict,
                "because": because,
                # The three that moved today's answer most, in words. Only ever
                # read alongside the verdict, which is gated on the evidence.
                "reasons": [
                    {**item, "sentence": explain_mod.in_words(item)}
                    for item in reasons[:3]
                ],
                "all_contributions": reasons,
                "symbol": symbol,
                "as_of": latest.index[-1].date().isoformat(),
                # How old the bar this was computed from actually is. A daily
                # signal built on a twelve-day-old close is not a signal, and
                # `as_of` alone was too quiet about it -- it read as a label
                # rather than as a warning.
                "stale_days": (dt.date.today() - latest.index[-1].date()).days,
                "close": (round(float(prices["close"].iloc[-1]), 4)
                          if prices is not None and len(prices) else None),
                "probability_up": round(probability, 4),
                # Confidence, not a recommendation. The strength is how far from
                # a coin flip the model is, and the UI shows it as that.
                "confidence": round(abs(probability - 0.5) * 2.0, 4),
                "leaning": "up" if probability > 0.5 else "down",
                "features": {name: round(float(value), 6)
                             for name, value in latest.iloc[0].items()},
            })
        except Exception as exc:                        # noqa: BLE001
            logger.warning("No signal for %s: %s", symbol, exc)

    return signals


# --- the news store --------------------------------------------------------

def record_paper(run: Run, *, top_n: int | None = None) -> dict | None:
    """Write what this run intends to do into the day-trade ledger.

    **No longer called.** That ledger records a book of 238 names rebalanced
    every session, which costs 96% of a 500 account in one round trip and ends
    at zero on day one -- see broker.py. The committed book in holding.py
    replaced it. Kept only so the record already written can still be read.

    Called once a run is done. Deliberately after scoring and deliberately
    one-way: `paper` is never imported by anything that builds a feature, and
    nothing here reads the ledger back. See paper.py for why that seam matters
    more than it looks.
    """
    try:
        return paper_mod.record_intent(run, top_n=top_n)
    except Exception as exc:                            # noqa: BLE001
        logger.warning("Could not record paper intent: %s", exc)
        return None


def follow_book(run=None, watchlist: Sequence[str] | None = None) -> dict:
    """Move the committed book forward one step: plan, fill, hold, or sell.

    The search committed one position rebalanced every sixty sessions, which is
    the only shape this account size can carry -- see holding.py. Called after a
    run finishes (to plan from today's signals) and on every scheduler cycle (to
    fill what was planned and to sell when the review session arrives).
    """
    try:
        if run is not None and (run.signals or []):
            holding_mod.plan_all(run)

        symbols = set(holding_mod.symbols_to_price())
        if not symbols:
            return {"note": "nothing planned or held"}

        frames = prices_mod.load_many(sorted(symbols), period="2y")
        stepped = holding_mod.advance_all(frames, run=run)

        # A review that sold leaves a book in cash with today's opinion still
        # in hand, so it can choose again in the same pass.
        if run is not None and any(s.get("exited") for s in stepped.values()):
            holding_mod.plan_all(run)

        funded = holding_mod.funded()
        return {**stepped.get(funded, {}), "books": {
            book_id: {k: v for k, v in step.items() if k != "holding"}
            for book_id, step in stepped.items()}}
    except Exception as exc:                            # noqa: BLE001
        logger.warning("Could not move the held book forward: %s", exc)
        return {"error": str(exc)}


def challenge_step(watchlist: Sequence[str] | None = None) -> dict:
    """Ask one new question of the search, if this week has budget left.

    Narrow and slow on purpose: see challenge.py. A qualifying challenger does
    not take over, it starts a forward record of its own.
    """
    try:
        symbols = universe_mod.resolve(watchlist) if watchlist else default_watchlist()
        waiting = challenge_mod.candidates()
        if not waiting or challenge_mod.spent_this_week() >= challenge_mod.BUDGET_PER_WEEK:
            return {"asked": None}

        frames = prices_mod.load_many(symbols, period="10y")
        return challenge_mod.step(frames)
    except Exception as exc:                            # noqa: BLE001
        logger.warning("Challenger pass failed: %s", exc)
        return {"asked": None, "error": str(exc)}


def promotion() -> dict:
    """What the promotion rule says today. Reading it changes nothing."""
    try:
        return promote_mod.decide()
    except Exception as exc:                            # noqa: BLE001
        logger.warning("Could not read the promotion rule: %s", exc)
        return {"promote": None, "why": str(exc)}


def write_briefing(watchlist: Sequence[str] | None = None) -> dict | None:
    """Say what changed today, from the stores. Once a day, and often nothing."""
    try:
        symbols = universe_mod.resolve(watchlist) if watchlist else default_watchlist()
        return briefing_mod.write(symbols)
    except Exception as exc:                            # noqa: BLE001
        logger.warning("Could not write the briefing: %s", exc)
        return None


def learn_names(watchlist: Sequence[str] | None = None) -> dict:
    """Ask what a few more tickers are called. Bounded, and runs before speak.

    Before this, the names the page printed came out of a language model's
    memory, which is a fine source right up until the ticker is obscure. A
    name beside its ticker in the material is something a reader can check.
    """
    try:
        symbols = universe_mod.resolve(watchlist) if watchlist else default_watchlist()
        # The names that actually appear on the page come first: the universe
        # is large and mostly silent, while a suggestion is a name somebody is
        # being invited to go and look at.
        wanted = list(names_mod.missing(symbols))
        try:
            opinion = attention_mod.opinion(symbols)
            front = [row["symbol"] for row in opinion.get("loudest") or []]
            front += [row["symbol"] for row in opinion.get("suggestions") or []]
            for gap in opinion.get("unwatched_markets") or []:
                front += gap.get("names") or []
            wanted = [s for s in front if s in set(names_mod.missing(front))] + wanted
        except Exception:                               # noqa: BLE001
            pass
        return names_mod.learn(dict.fromkeys(wanted))
    except Exception as exc:                            # noqa: BLE001
        logger.warning("Could not learn names: %s", exc)
        return {"asked": 0, "learned": 0}


def speak(watchlist: Sequence[str] | None = None) -> dict | None:
    """Let it say something, if the archive moved since it last spoke.

    Runs every cycle rather than once a day, because the point of this one is
    that somebody busy can read back what happened while they were. It is
    silent unless the findings changed, and silent entirely when no model is
    answering -- see chat.py for why both of those matter more than the
    feature does.
    """
    try:
        symbols = universe_mod.resolve(watchlist) if watchlist else default_watchlist()
        return chat_mod.on_cycle(symbols)
    except Exception as exc:                            # noqa: BLE001
        logger.warning("Could not write a note: %s", exc)
        return None


def guard_book() -> dict:
    """Stop following the funded book if it has stopped doing its job.

    Acts rather than reports, because a kill switch nobody pulls is a comment.
    It can only ever stop: nothing in the loop starts trading something.
    """
    try:
        return promote_mod.guard()
    except Exception as exc:                            # noqa: BLE001
        logger.warning("Could not read the drift guard: %s", exc)
        return {"stand_down": False, "why": str(exc)}


def settle_paper(watchlist: Sequence[str] | None = None) -> dict:
    """Fill and mark every paper entry whose session has now happened.

    The symbols come from the ledger, not from the watchlist: an entry holds
    what it holds, and settling a 238-name book against whichever universe the
    caller happened to pass drops the rest as "missing" and scales what is left
    up to a full book. That is how the first real settled day came to be 3.2%
    of its own intent.
    """
    symbols = sorted(paper_mod.pending_symbols())
    if not symbols:
        symbols = universe_mod.resolve(watchlist) if watchlist else default_watchlist()
    try:
        frames = prices_mod.load_many(symbols, period="2y")
        return paper_mod.settle(frames)
    except Exception as exc:                            # noqa: BLE001
        logger.warning("Could not settle the paper ledger: %s", exc)
        return {"settled": 0, "pending": 0, "error": str(exc)}


def collect_news(watchlist: Sequence[str] | None = None) -> dict:
    """One append-only pass over the news store. Called by the watcher.

    Separate from `collect_all` because it is on a different clock: models
    finish every few hours, news arrives all day, and the store is only worth
    anything if it is written to continuously from now on. See news.py for why
    it cannot be backfilled later.
    """
    symbols = universe_mod.resolve(watchlist) if watchlist else default_watchlist()
    added = news_mod.collect(symbols)
    return {"added": sum(added.values()),
            "symbols": len([s for s, n in added.items() if n]),
            "readiness": news_mod.readiness(symbols)}
