# Trader

Direction signals for US and European equities, trained two ways and scored
honestly.

It fetches daily prices, builds features from five sources, and trains on the
training half — here in numpy, or on HelloWorldAi across a network, or both at
once so the two can be compared. Either way the model is graded on the last
stretch of history it was never given.

**It produces an opinion about direction and nothing else.** No orders, no
broker credentials, no keys to anything that can spend money. Adding execution
later is a deliberate separate step, not a switch to flip.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt

cp env/.env.example env/.env      # then put a submitter key in it

python train.py                   # train here AND on HelloWorldAi, then compare
python train.py --backend local   # here only: ~30s, no network, no key needed
python search.py run --horizon 1,5   # try configurations; the test set stays sealed
python run.py                     # the UI, on http://127.0.0.1:8600
python watch.py --news            # collect finished models, grow the news store
```

The first run fetches a few hundred price histories and takes some minutes.
Everything after that is cached. Set `TRADER_UNIVERSE=core` to work with ten
symbols while iterating on code.

---

## What it actually found

Something more interesting than "no edge", and worse.

On 238 symbols with a relative target — 428,100 training rows, 107,165 test rows
it never saw — a plain logistic regression is **reliably more accurate than the
baseline and reliably loses money**:

| | |
|---|---|
| Accuracy, out of time | 51.11% |
| Baseline (the training majority) | 50.03% |
| Edge | **+1.08 points** |
| Rows / effective rows | 107,165 / 25,520 |
| Chance would produce | 0.63 points |
| Seed-to-seed spread | 0.07 points |
| Walk-forward windows positive | **6 of 6** (mean +1.47, sd 0.56) |
| | |
| Sharpe, after costs | **−0.68** |
| Return t-statistic | **−0.93** |
| Annualised | **−3.2%** |

Every accuracy hurdle passes, and passes comfortably. The edge is seventeen
times the seed spread, nearly twice what chance produces on the discounted
sample, and positive in *every one* of six consecutive out-of-time windows.
It is not noise.

It is also worth −3.2% a year, because the model is right about small moves and
wrong about large ones. That is a real and well-documented way to be accurate
and broke, and it is exactly what `evaluate.py` was always warning about:
*"Being right about nine small moves and wrong about one large one is a losing
week that reads as 90% accurate."*

**The gate opened on it.** Five hurdles were not enough, and I only found that
out by running them on a wide panel and looking at the money column. There is
now a sixth — the return's t-statistic has to clear 2 — and the page it would
have printed "Buy now" on now reads *still collecting data*.

That is the whole argument for building the harness before the strategy. The
harness is what tells you that your most convincing result is a losing one.

For contrast, the original absolute target on the same panel: the seed spread
alone was 1.84 points, walk-forward was positive in 1 of 6 windows, and the mean
edge across windows was −2.44%. Every "edge" ever measured on that setup was
smaller than the gap between a model and itself.

---

## The number that decides it: what an order could actually reach

The features on day *t* are computed from day *t*'s close. Nobody knows that
close until the session has ended, so nobody can be positioned **at** it on the
strength of it. The earliest an order can go in is the next open.

But the label runs close to close. So the thing the whole harness was grading is
a trade that cannot be placed.

Decomposing the gross P&L of the logistic control by holding window — same rows,
same positions, same demeaning, only the window differs:

| window | gross P&L per row | |
|---|---|---|
| close(t) → close(t+1) — **what the label means** | **+2.96 bp** | daily Sharpe **+1.67** |
| ↳ the overnight gap alone | **+3.84 bp** | *more than all of it* |
| ↳ open(t+1) → close(t+1) — **what an order reaches** | **−0.80 bp** | Sharpe **−0.40** |

The entire gross edge is the overnight gap, and the gap belongs to whoever was
already holding. By the open it has happened; what remains is negative **before
costs**. Zero commission would not save this.

That reframes the earlier finding. "Accurate but unprofitable after costs" was
true and too kind. The deeper problem is that the quantity being measured was
never reachable — and the most encouraging number in the project, a daily Sharpe
of +1.67, is the one that describes it.

So every evaluation now reports both windows, `labels.executable_return`
computes the second, and **the gate reads the executable one**. The verdict says
so in as many words when they disagree:

> +0.8% over the baseline, and close to close that is Sharpe +1.67 — but held
> from the first open after the signal exists it is −0.40, t −1.4. The edge is in
> the overnight gap, which is gone by the time anybody could trade on it.

Both series are demeaned the same way for a relative target, so the comparison
between them is a comparison of windows and not of conventions. Old runs without
the field fall back to the graded number, which is the optimistic assumption
this argument exists to stop anybody making silently.

---

## The paper ledger: the one measurement that cannot be mined

`/pnl` is a forward paper account — $500, filled at the first open after each
signal, closed at that session's close, 5bp charged each way. No orders are
placed and no broker is involved.

Everything else in this project is a backtest, and a backtest can be mined. This
cannot, because when each line is written **the outcome has not happened yet**.
That one property is worth more than any amount of careful splitting.

It is also fragile in exactly one way, so there is a rule:

> **The model never reads the ledger.** It is a scorecard, never an input.

A system that tunes itself on its own paper results turns the only un-mineable
measurement in the project into another training set — and you would spend
months of calendar time producing a number with the same flaw as the backtest,
except now you would believe it more because it came from "live" trading.
`dataset.py`, `features.py` and `labels.py` cannot import `paper.py`, and there
is a test that fails if they ever do.

What the system *may* do with its own results is **notice**: `divergence()`
compares the live Sharpe against what the backtest predicted and says whether
they have come apart, in standard errors. That is the honest version of "self
improving" — it tells you the model has stopped working, which is the thing
worth knowing, without letting it fit to the answer.

### Why it does not back its best five ideas

The obvious design — pick the most confident names and concentrate — was
measured on this panel, and it is backwards:

| top N | positions/day | close→close | | **open→close (real)** | |
|---|---|---|---|---|---|
| | | ann | Sharpe | **ann** | **Sharpe** |
| 1 | 2 | **+40.9%** | 1.13 | **−7.2%** | −0.15 |
| 5 | 10 | +26.0% | 1.38 | **−7.3%** | −0.48 |
| 10 | 20 | +30.6% | 1.94 | −1.2% | −0.05 |
| 119 | 238 | +17.0% | 2.98 | **+5.1%** | **1.14** |

Backing the top 5 shows **+26% a year** close-to-close and **−7.3%** over the
window an order can reach. The most confident calls are precisely the ones whose
edge sits in the overnight gap, so concentrating on them concentrates on the
part that is already gone by the open. Sharpe climbs monotonically with breadth
in *both* columns.

So `top_n` defaults to None — the whole book — and setting it is a decision the
P&L page will then show you the consequences of.

The ledger records even while the gate is shut, marked `shadow`. A forward
record of a model with no edge is exactly how you learn it still has none.

---

## Two trainers, and why that is the point

Training happens in one of three places, and the useful one is both:

| `--backend` | what it does |
|---|---|
| `local` | trains here in numpy — about 30 seconds, no network, no GPU, no key |
| `helloworld` | upload, submit, poll, download — the original path |
| `both` | the same rows to both, then compare them (**default**) |

**Your GPU is not the bottleneck, and neither was the network.** The model is
**7,233 parameters** — 46 → 64 → 64 → 1. Training it for the full step count is
26 seconds of numpy on a CPU; a full train-and-score cycle is about 30. At batch
size 64 an RTX 3070 is launching kernels for matrices too small to fill it. What
was being shipped across a network and queued behind a node-placement bug was
half a minute of arithmetic.

So the HelloWorldAi path stays, because running a real workload through a
distributed trainer is the point of having built one — but it is no longer the
only way to get a model, and it now has something to be checked against.

**Both backends produce the same artifact.** A locally trained network is packed
into the same zip of `model.safetensors` plus a `config.json` manifest that
HelloWorldAi returns, loaded by the same `model.load_bundle`, run through the
same numpy forward pass, and scored by the same evaluator on the same held-out
rows. Nothing downstream knows which one it got.

That identity is what makes the comparison mean anything: same rows, same
hyperparameters, same everything except where the gradient descent happened. Any
gap is a fact about the **round trip** — placement, their trainer, the holdout
it carves out, the weights that came back — rather than about the data.

And the gap has a scale. Two models of this shape on these rows differ by the
seed alone, and the noise floor already measures how much that is worth, so the
page reports the gap in multiples of it:

> The two agree to within the seed spread (1.62 points). The round trip is
> returning what training here returns, which is what it should do.

A gap several times the seed spread in HelloWorldAi's favour is treated as
*suspicious* rather than good news — both saw the same training rows, so a real
advantage has to have come from somewhere, and the usual somewhere is having
seen more than it should.

### The learning rate was killing the network

Running the local backend on the wide panel produced a model with an edge of
exactly **+0.0000** while a logistic regression on identical rows scored +1.08
points. An MLP with hidden layers strictly contains a logistic regression's
capacity, so that is not a fact about the data — it is a training failure.

It was, precisely:

```
up-rate 1.0000   accuracy 0.5003
probability range: 0.516992 .. 0.516992   std 0.000000
  layer 0: 64/64 units alive (0 dead)
  layer 1:  0/64 units alive (64 dead)
```

Every unit of the second hidden layer was dead. A ReLU whose pre-activation is
negative for every input outputs zero for every input *and has zero gradient*,
so it never recovers — the network had collapsed to its output bias and was
returning one constant probability for every row in the panel. Adam at
`lr=0.01`, on every seed tried.

`lr=0.01` was the default on **both** backends, and `pipeline` never overrode
it, so every HelloWorldAi run this project ever made was trained at it too.

| lr | accuracy | up-rate | dead in layer 2 |
|---|---|---|---|
| 0.01 | 0.4997 | 0.000 | **64 of 64** |
| 0.003 | 0.5054 | 0.875 | 38 of 64 |
| **0.001** | **0.5087** | **0.571** | **1 of 64** |
| 0.0003 | 0.5079 | 0.474 | 0 of 64 |

Now 0.001 — Adam's own default — everywhere. At full step count on the wide
panel the same model goes from an edge of +0.0000 to **+0.0091**, up-rate 1.000
to 0.556, probability std 0.000000 to 0.047.

The worst part is what the gate said about it: *"The model answers the same way
almost every day, so its accuracy is just the class balance. It has learned
nothing."* That is a sentence about the market, and it was a sentence about the
optimiser. So `train_local` now checks the network it produced and **refuses to
return a collapsed one**, naming the learning rate and saying in as many words
that nothing about the market can be concluded from it.

A training failure that reports itself as a market finding is the most expensive
kind of bug this project can have, because the conclusion it produces is exactly
the conclusion the project expects.

### And it stops itself now

With the learning rate fixed, the obvious next question is whether training
longer helps. It does not, and the curve is unambiguous:

| steps | passes | train acc | **test acc** | edge |
|---|---|---|---|---|
| 5,000 | 0.7 | 0.5194 | **0.5118** | +0.0115 |
| 20,000 | 3.0 | 0.5337 | 0.5108 | +0.0105 |
| 80,268 | 12.0 | 0.5564 | 0.5066 | +0.0063 |
| 200,000 | 29.9 | 0.5690 | **0.5029** | +0.0026 |

Training accuracy climbs the whole way; out-of-sample accuracy falls the whole
way. The model is not getting smarter, it is memorising — so looping training to
"keep improving" makes it strictly worse, and would have done nothing whatsoever
in the collapsed case, where the second layer had zero gradient from the start.

But picking 5,000 off that table means picking a hyperparameter by reading the
test set, which is how the one number that has to stay clean gets spent. So
`fit_mlp` holds back the **last 15% of the training rows** — chronologically, so
the slice is the most recent part of the training period — checks against it
every 1,000 steps, keeps the best weights, and gives up after 8 checks without
improvement.

It lands on 5,000 steps. The same place, chosen without looking:

| ceiling | stopped at | test acc | edge |
|---|---|---|---|
| 20,000 | 5,000 | 0.5077 | +0.0075 |
| 80,268 | 5,000 | 0.5077 | +0.0075 |
| 200,000 | 5,000 | 0.5077 | +0.0075 |

`steps` is now a ceiling rather than an instruction, a local run takes 6 seconds
instead of 69, and the step count stopped being a number anybody has to guess.

One more trap worth recording, because it would never have raised: the bundle format
ends in two class scores through a softmax and the local network ends in one
logit through a sigmoid. `softmax([a0, a1])[1]` is `sigmoid(a1 − a0)`, so the
"down" row must be **zero** and the "up" row the trained weights. Mirroring them
as `[−w, +w]` is the obvious thing to write, doubles the logit, and silently
sharpens every probability the model reports by up to 0.14. `train_local` loads
its own output back through the real loader and refuses to return a bundle that
disagrees with the network that produced it.

---

## What was wrong before, and what it cost

This is documented rather than quietly fixed, because each one produced numbers
that looked completely reasonable.

**A cached price history was never checked against the period requested.** A
short fetch of AAPL got cached early; every run afterwards asked for ten years,
got two, and never noticed. Apple had no rows before the cut date, so it was
silently dropped — nine symbols trained where ten were reported, for every run
in the project's history.

**The train/test split was re-derived at scoring time instead of being carried.**
It moved from 2024-11-06 to 2024-12-31, graded 4,187 rows while the run
advertised 4,100, and added a symbol to the test set that had been absent from
training. No rows leaked in that instance, but the direction of the drift was
luck: a single failed price fetch at scoring time changes the pool, which moves
the date, which can move it *earlier* and put trained rows into the score.

**Costs were charged per row rather than per trade.** The model held long on 99%
of days and changed position 113 times in 4,187 rows — and was charged 4,187
round trips.

```
gross                            +25.56% annualised
cost as coded (5bp every row)    +10.71%   <- what the UI reported
cost on actual turnover          +24.71%
```

That is 14.9 points of annual drag against a true 0.85. It turned a strategy
that essentially *was* buy-and-hold into one that appeared to destroy a sixth of
the account every year — an error in the direction that flattered the "no edge"
conclusion, which is still an error.

**RSI returned 100 through its entire warm-up.** The guard meant to blank it
tested `avg_gain.isna()`, which is true *during* the warm-up, so it kept the
fill instead of removing it. Harmless only because the 50-day average dropped
those rows anyway.

**Confidence buckets printed accuracy on five rows.** One stored run reads
"accuracy 1.0" on a bucket of five, with no warning. In a project whose whole
purpose is refusing to overclaim, that was the most misleading line on the page.

**Returns carried no uncertainty at all.** The gate computes a standard error
for accuracy and refuses to speak without one. The return figures printed next
to it — the numbers a person would actually act on — had none.

---

## The target is the most important decision here

Absolute direction — *will this close higher tomorrow* — is close to
unanswerable. About 52% of daily moves in a large-cap panel are up, so a model
that learns nothing and answers "up" every time scores 52%, and gradient descent
finds that constant long before it finds anything subtle. Measured here
repeatedly: up-rate 0.98, accuracy equal to the class balance, every feature
influence under 0.01. The model was not failing to learn. It had learned the
only thing reliably there, which is the drift.

Relative direction — *will this finish in the top half of its peers* — removes
the drift by construction. The classes are 50/50 on every date, so no constant
can score above chance, and the market factor that dominates absolute returns
cancels out of the target entirely.

Both targets are kept, because the comparison between them is informative. The
relative one produces a real and repeatable *accuracy* edge on a wide panel;
neither has produced a profitable one.

### How much evidence a panel is actually worth

Ten symbols on one day are not ten independent verdicts on a model, so the gate
discounts the row count by a design effect measured from the panel. All four
numbers below come from the same 238 symbols and the same 107,165 test rows:

| target | model | effective rows | design effect |
|---|---|---|---|
| absolute | always-up | 4,534 | **23.6** |
| absolute | logistic | 8,534 | 12.6 |
| relative | always-up | 107,165 | 1.0 |
| relative | logistic | 25,520 | 4.2 |

Two things fall out of that, and both matter more than they look.

**On the absolute target a day is essentially one observation.** Everything rises
and falls with the market, so being wrong about one name means being wrong about
nearly all of them: 107,165 rows are worth 4,534. That is roughly the number of
*trading days* in the test window. All the apparent power of a wide panel
evaporates, and a gate computed on the raw row count would be asking for about
a fifth of the evidence it thinks it is.

**The discount depends on the model as well as the target.** A constant answer
on a relative target has errors that are just the labels, which are balanced per
date by construction, so nothing correlates. A model that actually varies makes
*correlated mistakes* — when it misreads the market it misreads it in every name
at once — and four fifths of its sample goes. So the design effect is recomputed
for every model rather than assumed once for the panel.

The practical consequence: the relative target buys about **5.6× more
independent evidence** from the same rows. That is a bigger effect than anything
the feature engineering achieved.

---

## Five kinds of input

46 features today, from four blocks, each optional so its contribution can be
measured rather than assumed. A fifth -- the news store -- is wired in and
deliberately switched off until it has history; see below.

**The symbol's own prices** (18). The original nine were nearly the same feature
— returns over three horizons, price against two moving averages, all momentum
wearing different hats. A model given five correlated views of one thing has one
input, not five. What was added is information of a different kind: where in the
day's range it closed, how much of the move happened overnight, how far it sits
below its year's high, 12-1 momentum, return skew, and Amihud illiquidity.

**Where it sits among its peers** (9). Percentile ranks within the panel, plus
breadth and dispersion. Relative momentum is the oldest factor in the
literature; a rank is also how a name's own history gets normalised against a
market that was calm in 2017 and is not now.

**The state of the world** (16). This is the answer to "can we add economic
news", and it is deliberately not news — see below.

**Distance to the next announcement** (3). Post-earnings-announcement drift is
one of the few equity anomalies that has survived fifty years of people trying
to arbitrage it away, and it is visible at exactly this horizon. It costs two
integers and no scraping.

### Why macro is prices, not headlines

Scraping CPI headlines gives about **twenty observations** across a two-year test
set — twelve prints a year, identical across every symbol on the day they land.
Nothing can be learned from twenty points and nothing can be graded on them
either. A macro *event* feature is the smallest dataset in the project wearing
the costume of the largest.

The same information arrives continuously through prices: the curve steepens
before the print and after it, the dollar moves every session, credit widens
while the story is still forming. That is 2,500 observations rather than twenty,
it is the mechanism by which macro actually reaches equity prices, and it costs
one HTTP request per series.

It also avoids the worst leak in the macro family. **GDP and employment are
revised**, and FRED serves the current revision — so a model trained on "what GDP
was in March 2024" is trained on a number that did not exist in March 2024.
Getting that right needs vintage data (ALFRED, not FRED); getting it wrong is
invisible. A yield close is a yield close forever.

### Everything shared is lagged one session

The panel holds US and European names. The S&P closes at 16:00 New York; Nestlé
closes at 17:30 Zurich, which is 11:30 New York. Today's US macro close is five
and a half hours *after* the European bar it would be attached to — and even
within the US the VIX settles fifteen minutes after the equity close. The same
applies to cross-sectional ranks: ranking Nestlé against Apple on the same
calendar date compares closes taken hours apart.

So the macro, cross-sectional and news blocks are all shifted by one session,
uniformly. It costs a day of freshness and removes an entire class of leak that
would otherwise be invisible and would flatter every European name in the panel.

---

## What the model has to beat

Nine models were once trained on a GPU across a network and compared only
against the class balance. A logistic regression on the same rows takes four
tenths of a second and scored the same. That does not mean the distributed
training was broken — it means nobody could have told if it were.

Three controls now run locally before every submission and their scores are
stored with the run. They are trained with **exactly the hyperparameters being
submitted** -- same width, depth, batch size and step count -- because a control
trained for a different length than the model it is a control for is not
answering the question. Counting gradient steps rather than epochs also means a
240-symbol panel costs the same as a 10-symbol one, instead of twenty-four times
as much; the epoch-counting version took minutes on a wide panel, which is long
enough that somebody would turn the controls off.

- **Majority** — answer the training majority every time. The floor.
- **Logistic regression** — same rows, same split, same scaler. If the network
  does not beat this, the round trip bought a linear model slowly.
- **A local MLP of the same shape** — same width, depth and step count as the
  job being submitted. If the remote model scores meaningfully *worse* than
  this, the problem is in the round trip rather than in the data.

And two things the controls make possible that a single trained model cannot:

**Walk-forward.** Six consecutive out-of-time windows instead of one. One test
period is one draw, and a single flattering window is the most common way a
strategy that does not work comes to look as though it does. Six GPU round trips
would take a day; this takes seconds, and it is asking about the *data* rather
than about any particular trained network.

**A noise floor.** The same configuration trained several times with different
seeds. Whatever spread that produces is the smallest difference between two
models that means anything.

---

## The gate

`/analysis` is the trading view: one call per symbol, with the argument for it.
Three states — **Buy now**, **No buy**, **Still collecting data** — and one rule
that decides between them.

Every call is gated on the model having earned an opinion, and the bar is
computed rather than chosen. There are six hurdles:

1. **Enough distinct days.** A wide panel over three weeks is still three weeks.
2. **Enough *effective* rows.** Ten symbols on one day are not ten verdicts. The
   design effect is measured from the panel rather than assumed — about 1.7 on
   the original absolute-target panel, so the honest bar was a third higher than
   the one being used.
3. **The model varies.** A model answering one way every day has an accuracy
   that is just the class balance.
4. **It beats the baseline by more than chance**, where the baseline is the class
   that dominated *training* — the only one anybody had in advance. Using the
   test period's own majority was measuring against a number knowable only
   afterwards.
5. **The edge exceeds the noise floor**, and holds up in most walk-forward
   windows.
6. **The money is distinguishable from luck.** After costs, the strategy's
   return needs a t-statistic of 2. This is the hurdle the wide panel forced —
   see above; accuracy alone opened the gate on a model losing 3.2% a year.

Every hurdle is recorded whether it passed or not, and the page shows the whole
list. A gate that says no without saying which question it failed teaches nobody
anything.

When the gate is shut — which is every model this project has trained — every
symbol reads *still collecting data*, whatever the probability says.

I checked the gate rather than trusting it: 2,000 Monte Carlo trials of
skill-less models at up-rates from 0.50 to 0.995 opened it in 0.0–0.1% of cases.
The conservatism comes from the majority-class baseline being genuinely hard to
beat — a varying no-skill model loses to it by about 1.6 points on average.
That check was run at a one-day horizon, and it did not cover longer ones —
which is where the next section found a hole.

---

## Searching without spending the test set

A model trains in seconds now, so trying three hundred configurations is an
afternoon. What that afternoon spends is not compute, it is the test set: every
configuration scored against it is another question asked of the same rows, and
the best of three hundred answers is mostly the luckiest.

```bash
python search.py run --macro on,off --horizon 1,5,20 --note "does macro help at longer holds"
python search.py board                       # leaders on validation, and the bar they must clear
python search.py commit 7 --why "best validation Sharpe; simplest of the top three"
python search.py open                        # the sealed period, once
```

The panel is cut in three. The last 20% is sealed; of what is left, the last
20% is validation. `run` fits on the first part and scores on validation, with
the sealed rows truncated out before any model sees them — the search cannot
reach them by construction, and a test checks that. Every trial is appended to
`data/search/trials.jsonl`, and only validation numbers are ever written there.

`open` refuses until one configuration has been **committed in writing, with a
reason**. Deciding before looking is the difference between a test and a
search. A second opening is allowed — forbidding it would just mean deleting a
file — but it needs `--again`, it is counted, and the reading says plainly that
it is no longer a test.

**The trial count sets the bar.** Looking `n` times and keeping the best means
at least one false pass with probability 1 − 0.95ⁿ: 19% at four trials, 99% by
ninety. `board` and `open` both apply a Šidák correction and print what an edge
has to clear given how many times you have looked, sized by effective rows
rather than raw ones.

### The first dry run, and the bug it found

Four configurations on the ten-symbol core panel, in a throwaway ledger. The
winner — macro on, five-day horizon — scored **+2.5 points of edge, Sharpe
+2.52, t +3.05** on validation, and the board said it had earned a commit. On
the sealed period it scored **+0.13 points** and a negative Sharpe. Four tries
had been enough to find noise dressed as a strategy.

That was the seal doing its job. But a validation t of 3 on a signal that
vanished was worth explaining, and the explanation was not luck alone.

**Every statistic treated overlapping holding windows as independent days.** At
a five-day horizon, consecutive rows share four days of the same move, and a
real model's positions persist because the features behind them change slowly —
so it is graded on nearly the same return again and again. Measured on pure
noise with sticky positions, a model with no skill at all cleared t = 2 in
**36% of runs at five days and 56% at twenty**, against the 5% the threshold
promises. At one day it held at 5%, which is why nothing before had caught it.
Sharpe was separately annualised as though a five-day return were a daily one,
inflating it by √5. A search that varies the horizon would have climbed that
slope and called it a finding.

Now the rows carry their horizon (`Split.horizon`, set once from the spec), and
`evaluate` uses it: Sharpe is annualised over 252 / h periods, the t-statistic
and effective rows are corrected for overlap with Hansen–Hodrick long-run
variance at lag h − 1, and drawdown compounds a book that actually rebalances
every h sessions. The same noise test now gives **6.7% at five days and 6.3% at
twenty**. At one day every number is identical to before.

Rerun with the fix, the same winner reads Sharpe +1.13 and t +2.11, the
accuracy bar rises from +2.07 to +2.88 points, and **the board rejects it
before the test set is opened** — which is the order this is meant to happen in.

---

## Adding news, honestly

`news.py` is a point-in-time store, and the reason it looks so unlike a scraper
is that the scraping is the easy half and the timestamps are the hard one.

**A backfilled archive cannot be trusted.** Ask any provider today for news about
a symbol and you get what it has now, ranked by what it thinks matters now.
Stories that turned out to be important were promoted, stories that turned out
to be noise were dropped, and `published_at` is very often ingestion time rather
than when the item crossed the wire. A model trained on that has been shown
which stories mattered — which is precisely what it was supposed to work out —
and none of it is visible in the output.

The available source returns ten recent items per symbol. That is not a
limitation to work around; it is the shape of the honest problem. **There is no
archive to backfill from, so the store is built forwards.**

So the store records `captured_utc`, written by this machine at the moment the
item is appended and revisable by nobody. The provider's `published_utc` is kept
beside it and never used as a feature. The file is append-only, because its
entire value is that its earlier lines could not have been adjusted afterwards.

And news features stay **inert until the store has a year of history**.
`readiness` reports how much exists; the pipeline refuses the block until it
clears the threshold and says so. Turning it on over an empty store would feed
the model zeros for every historical row and a real number for today, which is
not a feature — it is a date stamp.

Start `python watch.py --news` now and the block becomes usable in about a year.
That is the real cost of doing this honestly, and it is worth knowing before
building rather than after.

---

## Asking it questions

`context.py` is the reading panel, and it is built so it cannot become the
trading half.

**It never touches the model or the score.** Nothing there feeds a feature, a
label or an evaluation — it reads the store and the finished run and writes
prose. That is the same seam already drawn around order execution, for the same
reason.

**It is gated by the same trust object as everything else.** A fluent paragraph
next to a probability is more dangerous than no paragraph, because fluency reads
as conviction. The page currently says *"still collecting data — this model has
not shown an edge yet"*, which is the most valuable sentence in the project, and
a confident-sounding narrative beside it would quietly undo that. When the gate
is shut the panel still renders, headed **Background — not evidence**.

**It retrieves and cites; it does not conclude.** Every claim traces to a stored
item with a capture timestamp. It is asked for what would make the reading wrong
as well as what supports it — the more useful half, and the less dangerous one.
It will not make a recommendation, because the only opinion in this project
comes from the gate, which is computed rather than written.

**Headlines are untrusted input.** They are scraped text written by strangers,
and anything in them that looks like an instruction is data. They go into the
request fenced and labelled, and the system prompt says so.

Optional: without `ANTHROPIC_API_KEY` and the `anthropic` package, the panel
shows its sources without prose and everything else works unchanged.

---

## How the pieces fit

```
universe.py   which symbols, and why width is the point
prices.py     daily OHLCV, split-adjusted, cached -- and the cache is now
   |          checked against the period asked for
features.py   18 per-symbol indicators, every one computable at that close
cross.py      9 -- where this name sits among its peers, lagged one session
macro.py      16 -- the state of the world, from unrevised market data, lagged
events.py     3 -- distance to the next announcement, clipped to what was known
news.py       a point-in-time store; inert as a feature until it has history
labels.py     the only forward-looking lines, kept separate; two targets
dataset.py    one cut date, carried not re-derived; purge; scaler on train only
   |
baseline.py   majority, logistic, local MLP, walk-forward, noise floor
trainer.py    two backends producing the same artifact, and what their
   |          disagreement means, in units of the seed spread
helloworld.py upload -> submit -> poll -> download
   |
model.py      numpy forward pass; no torch, no GPU, no black box
evaluate.py   accuracy vs a training-period baseline, turnover-based costs,
   |          Sharpe, t, drawdown, effective sample size -- all horizon-aware
explain.py    the six-hurdle gate, and per-day attribution underneath it
search.py     train / validation / sealed test, a trial ledger, commit-then-open
context.py    prose and citations, downstream of everything, gated
web/          the UI; watch.py does the collecting unattended
```

---

## Limitations worth knowing

**The universe is survivor-biased.** The symbol lists are liquid large caps with
long histories, not index constituents as of any particular date. A name that
left an index or was acquired is not there. Fixing it properly needs a
point-in-time constituent history, which is a paid dataset.

**Earnings coverage is uneven.** US names come back with a decade of quarterly
dates; some European names come back with three. Symbols below the threshold get
neutral values and are named in the coverage report rather than silently filled.

**The one-session lag is blunt.** Grouping the panel by trading session and
ranking within each would recover a day of freshness. It is worth doing once the
universe is wide enough for the groups to be large.

**The accuracy edge is real and does not pay.** On the wide relative panel it
survives every statistical test in the project and still loses money after
costs. Nothing here is tradeable; that is the finding, not a caveat.

**The accuracy hurdle is still slightly generous.** Its standard error counts
the noise in the accuracy but not in the baseline, which is measured on the same
test rows. In the noise tests above, a skill-less model at one day cleared the
accuracy check several times more often than the roughly 2% that two standard
errors, one-sided, should allow.
The other hurdles — noise floor, walk-forward agreement, return t-statistic —
still stand behind it, which is why it has never opened the gate, but it is not
the bar it claims to be.

**Costs past one session are approximate.** Turnover is still charged day to
day on overlapping positions, where a book that rebalances every h sessions
would trade less often and by more each time.

**Only the local controls have been run at width.** The numbers above come from
the logistic and MLP controls on 238 symbols. No HelloWorldAi model has yet been
trained on a panel that size — the step count that panel earns is about twenty
times the old default, which is a real request of somebody else's GPU.

---

## Tests

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```

204 tests. The look-ahead ones test the property rather than the implementation
— features computed on a truncated history must match the full one — and there
is a test that deliberately introduces a centred rolling window to confirm the
property test can still fail. The macro, cross-sectional and event blocks each
have their own leak test. The model-loading ones build a bundle with
HelloWorldAi's own packing code rather than a fixture, so a format change fails
here instead of in production.

Several are regression tests for the bugs listed above: the pinned cut date, the
RSI warm-up, per-trade costs, the withheld small-bucket accuracy, the
training-period baseline, the append-only news store, the noise floor, and the
sigmoid-to-softmax conversion that would have sharpened every probability by up
to 0.14 without raising anything, the executable return that turns an
untradeable backtest into the number the gate reads, and the overlap correction
— which checks both that a skill-less model at five days used to clear t = 2
more than a quarter of the time, and that it no longer does.
