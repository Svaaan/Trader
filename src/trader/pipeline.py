"""The whole loop: build a dataset, send it, wait, collect, evaluate, signal.

A *run* is one pass through that, and it lives in a directory under data/runs
with everything needed to explain itself afterwards: what went in, what came
back, and how it scored. Nothing is held only in memory, because the interesting
part happens minutes or hours after the submit and the answer to "what did it do
and why" has to survive a restart.

Two things happen before anything is sent, and both exist because of what the
project measured about itself:

**The local controls run first.** A majority-class model, a logistic regression
and a small MLP trained with exactly the hyperparameters about to be submitted,
all on the same rows. Nine models were once trained on a GPU across a network
and compared only against the class balance -- a linear model would have scored
the same, and nobody could have known. Their scores go into the run so that the
returned model is always read next to something.

**The cut date is chosen once and written down.** Scoring passes it back rather
than re-deriving it. Re-deriving it moved the split by seven weeks, graded 87
rows the run never advertised, and quietly added a symbol to the test set that
had been missing from training.

The hand-off from HelloWorldAi is a poll, not a callback. The coordinator has no
way to call back into this project -- and a webhook would mean exposing a port
from a laptop, which is a worse trade than asking every thirty seconds.
`collect` is safe to call repeatedly; it picks up whatever has finished since
the last time and leaves the rest alone.
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
from . import news as news_mod
from . import prices as prices_mod
from . import trainer as trainer_mod
from . import universe as universe_mod
from .helloworld import Client, HelloWorldError

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

    The number still matters for the remote backend, which does its own
    training and does not stop itself.

    Floored, because a very small panel still needs enough steps to converge,
    and capped so that a very wide one does not queue on somebody's GPU for a
    day.

    Worth knowing before the first wide submission: this asks for roughly twenty
    times the old fixed 4,000 on a 240-symbol panel. Locally that costs nothing,
    because training stops when it stops helping; remotely it is a real request
    of somebody else's machine, and a coordinator with a per-job ceiling may
    refuse it. Pass `steps=` explicitly to override.
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
    task_id: Optional[str] = None
    status: str = "building"
    # Which trainer produced the headline numbers, and which were asked for.
    backend: str = trainer_mod.HELLOWORLD
    primary: str = ""
    spec: dict = dataclasses.field(default_factory=dict)
    dataset: dict = dataclasses.field(default_factory=dict)
    verification: dict = dataclasses.field(default_factory=dict)
    evaluation: dict = dataclasses.field(default_factory=dict)
    controls: dict = dataclasses.field(default_factory=dict)
    walk_forward: dict = dataclasses.field(default_factory=dict)
    # The local reference model, when one was trained. Kept beside rather than
    # inside `evaluation`, because the pair is the point -- see trainer.py.
    local_evaluation: dict = dataclasses.field(default_factory=dict)
    local_verdict: str = ""
    comparison: dict = dataclasses.field(default_factory=dict)
    verdict: str = ""
    signals: list = dataclasses.field(default_factory=list)
    trust: dict = dataclasses.field(default_factory=dict)
    learnt: list = dataclasses.field(default_factory=list)
    error: str = ""

    def save(self) -> None:
        directory = _run_dir(self.run_id)
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, STATE_FILE), "w", encoding="utf-8") as fh:
            json.dump(dataclasses.asdict(self), fh, indent=2)

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
        return os.path.join(_run_dir(self.run_id), BUNDLE_FILE)

    @property
    def local_bundle_path(self) -> str:
        return os.path.join(_run_dir(self.run_id), LOCAL_BUNDLE_FILE)

    @property
    def has_model(self) -> bool:
        return os.path.exists(self.bundle_path)

    @property
    def has_local_model(self) -> bool:
        return os.path.exists(self.local_bundle_path)

    @property
    def wants_remote(self) -> bool:
        return self.backend in (trainer_mod.HELLOWORLD, trainer_mod.BOTH)

    @property
    def wants_local(self) -> bool:
        return self.backend in (trainer_mod.LOCAL, trainer_mod.BOTH)

    @property
    def cut_date(self) -> Optional[pd.Timestamp]:
        """The split this run was built on. Scoring must reuse it."""
        raw = (self.dataset or {}).get("cut_date")
        return pd.Timestamp(raw) if raw else None


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
    return runs


# --- sending ---------------------------------------------------------------

def start(watchlist: Sequence[str] | None = None, *, horizon: int = 1,
          period: str = "10y", test_fraction: float = 0.2,
          target: str = labels_mod.RELATIVE,
          spec: dataset_mod.Spec | None = None,
          backend: str = trainer_mod.BOTH,
          steps: int | None = None, client: Client | None = None,
          run_controls: bool = True, folds: int = 6) -> Run:
    """Build a dataset from live prices and train a model on it.

    The test half never leaves this machine. HelloWorldAi gets the training rows
    only, so the score this project reports is measured on data no model in the
    chain has ever seen -- including through the coordinator's own verification,
    which holds back a random slice of whatever it is given.

    Pass `spec` to choose which feature blocks go in -- switching one off and
    re-running is how its contribution gets measured rather than assumed. The
    keyword arguments are the common case and are ignored when `spec` is given.

    `steps` defaults to whatever the dataset size deserves -- see `steps_for`.
    A fixed step count is a fixed number of *samples*, which is a shrinking
    number of passes as the panel widens, and an undertrained model looks
    exactly like a model with nothing to learn.

    `backend` chooses where the training happens:

      "helloworld"  upload, submit, poll, download -- the original path
      "local"       train here in numpy, about half a minute, no network
      "both"        do both on the same rows and compare them

    "both" is the one worth running. The two get identical rows and identical
    hyperparameters, so any gap between their scores is a fact about the round
    trip rather than about the data -- see trainer.py. A local-only run finishes
    before this function returns; anything involving HelloWorldAi comes back
    "training" and is picked up by `collect`.
    """
    if backend not in trainer_mod.BACKENDS:
        raise ValueError(
            f"backend must be one of {trainer_mod.BACKENDS}, not {backend!r}")

    client = client or Client()
    symbols = universe_mod.resolve(watchlist) if watchlist else default_watchlist()

    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    spec = spec or dataset_mod.Spec(target=target, horizon=horizon,
                                    test_fraction=test_fraction)

    run = Run(run_id=run_id,
              created=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              watchlist=symbols, horizon=horizon, spec=spec.to_dict(),
              backend=backend)
    run.save()

    try:
        frames = prices_mod.load_many(symbols, period=period)
        if not frames:
            raise ValueError("no price history could be fetched for any symbol")

        splits, cut, report = dataset_mod.build_panel(frames, spec)
        x_train, y_train, scaler = dataset_mod.combine(
            splits, report["feature_names"])

        description = dataset_mod.describe(splits, scaler, spec, report, cut)
        run.dataset = description

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
                run.controls = baseline_mod.run_controls(
                    splits, report["feature_names"], hidden=64, depth=2,
                    steps=training_steps, batch=BATCH_SIZE)
                run.walk_forward = baseline_mod.walk_forward(
                    frames, spec, folds=folds)
                run.save()
            except Exception as exc:                    # noqa: BLE001
                logger.warning("Controls failed, continuing: %s", exc)
                run.controls = {"error": str(exc)}

        hyper = trainer_mod.Hyperparameters(
            hidden=64, depth=2, steps=training_steps, batch=BATCH_SIZE)
        description["hyperparameters"] = hyper.to_dict()

        # --- train here, if asked ------------------------------------------
        #
        # Done first, and synchronously. It takes about half a minute, so a
        # `both` run has a scored model before the submit has finished
        # uploading -- which means the page has something to show during the
        # hour the remote job spends queued, and means a broken round trip is
        # visible as a gap rather than as an absence.
        if run.wants_local:
            bundle = trainer_mod.train_local(
                x_train, y_train, report["feature_names"], hyper)
            with open(run.local_bundle_path, "wb") as handle:
                handle.write(bundle)
            logger.info("Run %s: trained locally (%d bytes)", run_id, len(bundle))

        # --- and send it, if asked -----------------------------------------
        if run.wants_remote:
            blob = dataset_mod.pack_for_helloworld(x_train, y_train)
            description["bytes_sent"] = len(blob)

            artifact_id = client.upload_dataset(blob)

            # Choose the machine rather than letting the coordinator choose on
            # a flag that goes stale -- see Client.pick_node. None falls back to
            # its placement, which is right when every node is reporting
            # normally.
            node_id = client.pick_node()
            description["node_id"] = node_id

            run.task_id = client.submit(
                dataset_id=artifact_id,
                model_name=f"trader-{run_id}",
                steps=training_steps,
                batch_size=BATCH_SIZE,
                hidden_dim=64,
                depth=2,
                node_id=node_id,
            )
            run.status = "training"
            logger.info("Run %s submitted as %s", run_id, run.task_id)

        # A local-only run has nothing to wait for, so it finishes here.
        if run.wants_local and not run.wants_remote:
            _process(run)
            run.status = "done"
            logger.info("Run %s finished locally: %s", run_id, run.verdict)
        elif run.wants_local:
            # Score the local model now so the page is readable while the
            # remote job queues. `collect` re-scores both together when it
            # lands, which is where the comparison gets made.
            try:
                _process(run)
            except Exception as exc:                    # noqa: BLE001
                logger.warning("Could not score the local model yet: %s", exc)
            run.status = "training"

    except Exception as exc:                            # noqa: BLE001
        run.status = "failed"
        run.error = str(exc)
        logger.exception("Run %s could not be started", run_id)

    run.save()
    return run


# --- collecting ------------------------------------------------------------

def collect(run: Run, *, client: Client | None = None) -> Run:
    """Fetch and process the remote model if the job has finished.

    Safe to repeat. A run with no remote half has nothing here to wait for --
    it was finished by `start` -- so it is left alone rather than being polled
    for a task that does not exist.
    """
    if run.status in ("done", "failed") or not run.task_id:
        return run

    client = client or Client()

    try:
        job = client.job(run.task_id)
    except HelloWorldError as exc:
        logger.warning("Could not check %s: %s", run.task_id, exc)
        return run

    if job is None:
        run.status = "failed"
        run.error = f"{run.task_id} is not in this key's job list any more"
        run.save()
        return run

    run.status = {"pending": "queued", "running": "training"}.get(job.status, job.status)
    run.verification = job.verification

    if not job.finished:
        run.save()
        return run

    if not job.succeeded:
        run.status = "failed"
        run.error = (job.raw.get("result") or job.raw.get("error")
                     or f"the job ended as {job.status}")
        run.save()
        return run

    try:
        blob = client.download_bundle(run.task_id)
        with open(run.bundle_path, "wb") as handle:
            handle.write(blob)
        logger.info("Run %s: model saved (%d bytes)", run.run_id, len(blob))

        _process(run)
        run.status = "done"

    except Exception as exc:                            # noqa: BLE001
        run.status = "failed"
        run.error = f"the model came back but could not be used: {exc}"
        logger.exception("Run %s failed after download", run.run_id)

    run.save()
    return run


def collect_all(*, client: Client | None = None) -> list:
    """One pass over every unfinished run. What the watcher calls."""
    client = client or Client()
    return [collect(run, client=client) for run in list_runs()
            if run.status not in ("done", "failed")]


# --- what to do with a finished model --------------------------------------

def _process(run: Run) -> None:
    """Score whatever models this run has, out of time, on one rebuild.

    A run can hold two: the one HelloWorldAi returned and the one trained here
    on the same rows with the same hyperparameters. Both go through the same
    loader, the same forward pass and the same evaluator, because the only way
    the comparison between them means anything is if nothing else differs.

    The headline numbers describe the *remote* model when there is one -- it is
    the thing under test -- and the local one otherwise, so that a `both` run is
    readable during the hour it spends waiting rather than blank.
    """
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
    splits, report = dataset_mod.split_at(
        dataset_mod.prepare(frames, spec), cut)

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
    (x_test, y_test, returns, dates), symbols = dataset_mod.test_matrix(
        splits, scaler)
    train_up_share = (run.dataset.get("train") or {}).get("up_share")

    def score(path: str) -> tuple:
        model = model_mod.load_bundle_file(path)
        probabilities = model.probabilities(x_test)[:, 1]
        result = evaluate_mod.evaluate(
            probabilities, y_test, returns, dates, symbols,
            train_up_share=train_up_share)
        return model, result

    # --- the local reference, when there is one ---
    local_model = None
    if run.has_local_model:
        local_model, local_result = score(run.local_bundle_path)
        run.local_evaluation = local_result.to_dict()
        run.local_verdict = evaluate_mod.verdict(local_result)

    # --- the model under test ---
    if run.has_model:
        model, result = score(run.bundle_path)
        run.primary = trainer_mod.HELLOWORLD
    elif local_model is not None:
        model, result = local_model, local_result
        run.primary = trainer_mod.LOCAL
    else:
        raise ValueError("this run has no model to score")

    run.evaluation = result.to_dict()
    run.verdict = evaluate_mod.verdict(result)

    # --- what the gap between them means -------------------------------------
    #
    # Same architecture, same rows, same hyperparameters: the only thing that
    # should separate them is the seed, and the noise floor is how much that is
    # worth. Anything larger is the round trip doing something.
    if run.has_model and run.local_evaluation:
        run.comparison = trainer_mod.compare(
            run.local_evaluation, run.evaluation,
            noise_floor=(run.controls.get("noise_floor") or {}).get("spread"))

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
