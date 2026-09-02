"""The whole loop: build a dataset, send it, wait, collect, evaluate, signal.

A *run* is one pass through that, and it lives in a directory under data/runs
with everything needed to explain itself afterwards: what went in, what came
back, and how it scored. Nothing is held only in memory, because the interesting
part happens minutes or hours after the submit and the answer to "what did it do
and why" has to survive a restart.

The hand-off from HelloWorldAi is a poll, not a callback. The coordinator has no
way to call back into this project -- and a webhook would mean exposing a port
from a laptop, which is a worse trade than asking every thirty seconds. `collect`
is safe to call repeatedly; it picks up whatever has finished since the last
time and leaves the rest alone.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import os
from typing import Optional, Sequence

import numpy as np

from . import dataset as dataset_mod
from . import evaluate as evaluate_mod
from . import explain as explain_mod
from . import features as features_mod
from . import model as model_mod
from . import prices as prices_mod
from .helloworld import Client, HelloWorldError

logger = logging.getLogger(__name__)

RUNS_DIR = os.environ.get(
    "TRADER_RUNS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "runs"),
)

# Large caps on both sides of the Atlantic, liquid enough that a daily close is
# a real price rather than the last trade somebody happened to make.
DEFAULT_WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "JPM", "XOM",          # US
    "ASML.AS", "SAP.DE", "NESN.SW", "MC.PA", "VOLV-B.ST",   # Europe
]

STATE_FILE = "run.json"
BUNDLE_FILE = "model.zip"


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
    dataset: dict = dataclasses.field(default_factory=dict)
    verification: dict = dataclasses.field(default_factory=dict)
    evaluation: dict = dataclasses.field(default_factory=dict)
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
            return cls(**json.load(fh))

    @property
    def bundle_path(self) -> str:
        return os.path.join(_run_dir(self.run_id), BUNDLE_FILE)

    @property
    def has_model(self) -> bool:
        return os.path.exists(self.bundle_path)


def list_runs() -> list[Run]:
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
          steps: int = 4000, client: Client | None = None) -> Run:
    """Build a dataset from live prices and send it to HelloWorldAi.

    The test half never leaves this machine. HelloWorldAi gets the training rows
    only, so the score this project reports is measured on data no model in the
    chain has ever seen -- including through the coordinator's own verification,
    which holds back a random slice of whatever it is given.
    """
    client = client or Client()
    symbols = list(watchlist or DEFAULT_WATCHLIST)

    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run = Run(run_id=run_id,
              created=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              watchlist=symbols, horizon=horizon)
    run.save()

    try:
        frames = prices_mod.load_many(symbols, period=period)
        if not frames:
            raise ValueError("no price history could be fetched for any symbol")

        splits, cut = dataset_mod.build_panel(
            frames, horizon=horizon, test_fraction=test_fraction)
        x_train, y_train, scaler = dataset_mod.combine(splits)

        description = dataset_mod.describe(splits, scaler)
        description["cut_date"] = cut.date().isoformat()
        description["horizon"] = horizon
        run.dataset = description

        blob = dataset_mod.pack_for_helloworld(x_train, y_train)
        description["bytes_sent"] = len(blob)

        artifact_id = client.upload_dataset(blob)

        # Choose the machine rather than letting the coordinator choose on a
        # flag that goes stale -- see Client.pick_node. None falls back to its
        # placement, which is right when every node is reporting normally.
        node_id = client.pick_node()
        description["node_id"] = node_id

        run.task_id = client.submit(
            dataset_id=artifact_id,
            model_name=f"trader-{run_id}",
            steps=steps,
            hidden_dim=64,
            depth=2,
            node_id=node_id,
        )
        run.status = "training"
        logger.info("Run %s submitted as %s", run_id, run.task_id)

    except Exception as exc:                            # noqa: BLE001
        run.status = "failed"
        run.error = str(exc)
        logger.exception("Run %s could not be submitted", run_id)

    run.save()
    return run


# --- collecting ------------------------------------------------------------

def collect(run: Run, *, client: Client | None = None) -> Run:
    """Fetch and process the model if the job has finished. Safe to repeat.

    This is the automatic half: called on a timer, it picks up whatever has
    completed since the last pass and leaves everything else untouched.
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


def collect_all(*, client: Client | None = None) -> list[Run]:
    """One pass over every unfinished run. What the watcher calls."""
    client = client or Client()
    return [collect(run, client=client) for run in list_runs()
            if run.status not in ("done", "failed")]


# --- what to do with a finished model --------------------------------------

def _process(run: Run) -> None:
    """Score the model out of time, then read today's signal from it."""
    model = model_mod.load_bundle_file(run.bundle_path)
    scaler = dataset_mod.Scaler.from_dict(run.dataset["scaler"])

    frames = prices_mod.load_many(run.watchlist,
                                  period="10y")
    splits, _ = dataset_mod.build_panel(
        frames, horizon=run.horizon,
        test_fraction=1.0 - _train_fraction(run))

    # --- the honest score: rows that were never sent anywhere ---
    x_test = np.concatenate([s.x_test for s in splits])
    y_test = np.concatenate([s.y_test for s in splits])
    returns = np.concatenate([s.forward_returns_test for s in splits])

    probabilities = model.probabilities(scaler.apply(x_test))[:, 1]
    result = evaluate_mod.evaluate(probabilities, y_test, returns)

    run.evaluation = result.to_dict()
    run.verdict = evaluate_mod.verdict(result)

    # Whether anything below is worth printing. Decided once, from the
    # out-of-time score, and every per-symbol call is gated on it.
    trust = explain_mod.assess(run.evaluation)
    run.trust = trust.to_dict()

    # What the model attends to in general, which is how a network that has
    # collapsed to one answer shows itself from the inside.
    run.learnt = explain_mod.what_it_learnt(model, scaler, x_test)

    # --- and what it says about today ---
    run.signals = _todays_signals(model, scaler, frames, trust)


def _train_fraction(run: Run) -> float:
    train = run.dataset.get("train", {}).get("rows", 0)
    test = run.dataset.get("test", {}).get("rows", 0)
    total = train + test
    return (train / total) if total else 0.8


def _todays_signals(model, scaler, frames: dict, trust=None) -> list[dict]:
    """The most recent finished session, per symbol.

    This is the row with features and no label -- the one prediction exists for.
    """
    signals = []

    for symbol, prices in sorted(frames.items()):
        try:
            x_frame = features_mod.build(prices)
            if x_frame.empty:
                continue

            latest = x_frame.iloc[[-1]]
            row = latest[features_mod.FEATURE_NAMES].to_numpy(dtype=np.float32)
            probability = float(model.probabilities(scaler.apply(row))[0, 1])

            verdict, because = (explain_mod.rank(probability, trust)
                                if trust is not None else (explain_mod.UNSURE, ""))

            reasons = explain_mod.contributions(model, scaler, row)

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
                "close": round(float(prices["close"].iloc[-1]), 4),
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
