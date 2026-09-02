# Trader

Direction signals for US and European equities, trained on HelloWorldAi and
scored honestly.

It fetches daily prices, builds features, sends the training half to
HelloWorldAi to train on a real GPU, fetches the finished model back
automatically, and grades it on the last stretch of history it was never given.

**It produces an opinion about direction and nothing else.** No orders, no
broker credentials, no keys to anything that can spend money. Adding execution
later is a deliberate separate step, not a switch to flip.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt

cp env/.env.example env/.env      # then put a submitter key in it
python run.py                     # the UI, on http://127.0.0.1:8600
python watch.py                   # optional: collect finished models unattended
```

---

## What it actually found

The first real run, on ten large caps and ten years of daily closes, trained on
your RTX 3070 through HelloWorldAi:

| | |
|---|---|
| Accuracy, out of time | 51.7% |
| Baseline (always guess the commoner direction) | 51.6% |
| Edge | **+0.1%** |
| Days it said "up" | 98.8% |
| Strategy, annualised, after costs | +8.9% |
| Simply holding, same days | +22.8% |

The model learned to say "up" almost every day, which on a panel that rose 52%
of the time scores 51.7%. That is not a signal, and the UI says so in those
words.

This is the expected outcome and the reason the project is built the way it is.
Daily direction is close to a coin flip; the value here is a harness that can
tell you so rather than one that flatters a model until you believe it.

### The number that makes the point

HelloWorldAi's own verification scored the same model at **54.2%** against an
untrained floor of 50.7%, and that check is sound — the coordinator rebuilds the
model from the returned weights and scores it itself, so a node cannot report a
result it did not get.

But it holds back a **random** 20%. For a price series that means training on
Tuesday and Thursday and being graded on Wednesday, with both neighbours
memorised. Markets are autocorrelated enough for that alone to be worth a couple
of points.

Out of time, the same weights score 51.7% against a 51.6% baseline.

**54.2% and 51.7% are the same model.** The gap is the whole reason this project
keeps its own test set and never sends it anywhere. Both numbers are shown side
by side in the UI, because either one alone misleads.

---

## How the pieces fit

```
prices.py     daily OHLCV, split-adjusted, cached as CSV you can read
   |          drops today's unfinished bar -- it is not knowable yet
features.py   9 indicators, every one computable at that day's close
labels.py     the only forward-looking line in the project, kept separate
dataset.py    one cut date across all symbols; scaler fitted on train only
   |
helloworld.py upload -> submit -> poll -> download
   |
model.py      numpy forward pass; no torch, no GPU, no black box
evaluate.py   accuracy against baseline, by confidence, after costs
   |
web/          the UI; watch.py does the same collecting unattended
```

### Three things it is careful about

**Look-ahead.** Features may only look backwards, labels must look forwards, and
they live in separate files so the two cannot be confused. `tests/` checks the
property rather than the code: features are computed on a truncated history and
again on the full one, and every shared date must match. A centred rolling
window — the classic accidental leak — fails that immediately, which was
confirmed by introducing one on purpose.

**The split is one date across the whole panel.** Splitting each symbol at its
own 80% mark looks chronological and is not: symbols have different amounts of
history, so on the first real build Apple's training rows ran to 2026-04 while
SAP's test rows began in 2024-09. Twenty months of training on one company and
grading on another over the same days. These markets move together, so that is a
random split with extra steps.

**The scaler never sees the test period.** Standardising on the whole series
tells the model what the future looked like. Mean and standard deviation come
from training rows, are applied to both, and are stored with the run so a model
loaded months later uses the same numbers.

---

## Talking to HelloWorldAi

Four calls: upload the dataset, submit a job, poll `/my-tasks`, download the
bundle. The hand-off is a poll rather than a callback, because the coordinator
cannot reach into a laptop and opening a port to let it would be a worse trade.

`watch.py` and the UI both call the same idempotent `collect_all()`, so running
both, or neither, or restarting mid-job, all end in the same place.

### It picks the node itself

HelloWorldAi's automatic placement trusts each node's `isConnected` flag, and
that flag is not reconciled against the heartbeat. On the first real submission
the job was assigned to a node last heard from ten minutes earlier and sat
`pending` indefinitely — the stale-task reaper only rescues jobs that reached
`running`.

So `Client.pick_node()` chooses by the one field that cannot quietly go stale:
how long ago the node actually spoke. Falls back to the coordinator's placement
when nothing looks stale.

### No torch

The bundle carries a manifest listing the layers explicitly, so the model is
rebuilt from that and run with a few matrix multiplications in `model.py`. Two
gigabytes of PyTorch to push nine numbers through two hidden layers would also
have put a black box in the one place this project most needs to be readable.

It refuses what it does not recognise — a transformer bundle, an unknown layer —
rather than producing numbers that look plausible.

---

## Adding execution later

The seam is deliberate. Signals come out of `pipeline._todays_signals` as
probabilities with a confidence, and nothing consumes them but the UI.

Anything that places orders should be a separate module with its own
credentials, its own kill switch, and paper trading proven over months first —
and on this evidence there is nothing here worth trading. Building the harness
before the strategy is the right order; the harness is what tells you whether
there is a strategy.

---

## Tests

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```

22 tests. The look-ahead ones test the property, not the implementation, and
were checked against a deliberately introduced leak. The model-loading ones
build a bundle with HelloWorldAi's own packing code rather than a fixture I
wrote, so a format change fails here instead of in production; they skip if
HelloWorldAi is not checked out beside this project.
