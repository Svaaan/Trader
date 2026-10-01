# Trader

Direction signals for 241 US and European equities, graded honestly enough to
say when they are worthless — which, so far, is most of the time.

**It produces an opinion and a paper record. Nothing else.** No orders, no
broker credentials, no keys to anything that can spend money.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
cp env/.env.example env/.env          # optional: everything has a default

python train.py                       # train and grade, ~1 min
python run.py                         # the UI on http://127.0.0.1:8600
python auto.py                        # the scheduler: news, books, contest, briefing
python search.py board                # what has been tried, and the bar it must clear
```

First run fetches a few hundred price histories; everything after that is
cached. `TRADER_UNIVERSE=core` works with ten symbols while iterating.

---

## What it found

**The model is accurate and unprofitable.** On 464 sealed sessions it scores
50.8% against a 50.0% baseline — a real edge, positive in six of six
walk-forward windows — and it **loses money**, with an open-to-close
t-statistic of −3.19.

Both things are true because of *when* the move happens:

| window | per row | annualised |
|---|---|---|
| close → close, what the label describes | +2.5 bp | +14.4% |
| ├ overnight, close → next open | **+2.2 bp** | — |
| └ open → close, what an order can reach | +0.3 bp | +5.4% |

**Roughly 90% of the gross edge is the overnight gap**, which a signal computed
from the closing price cannot be positioned for. Every number here is reported
over both windows for that reason, and the gate reads the second one.

Trading the gap directly does not rescue it: the overnight drift is real
(+4.5 bp a night, t +3.77, 71% of symbols positive) and **smaller than the
spread you cross to capture it** — 10 bp there and back, before commission.

## What a trade costs, and why that decides the design

A retail Nordic broker charges a percentage **or a minimum, whichever is
larger**. Read off their own price pages on 2026-09-28 (`broker.py` keeps each
source and date):

| | percentage | minimum per side | FX |
|---|---|---|---|
| Avanza Start, Stockholm | 0% | 0 kr | — |
| Avanza Mini, US | 0.25% | $1 | 0.25% each way |
| Nordnet Mini, outside the Nordics | 0.25% | 9 kr | 0.25% each way |

On a $500 account spread over 238 names, a position is **$2.10** and a round
trip costs **$2.01 — 96% of the position**. The first day of that book costs
$478 and there is no second day.

So the shape is not a tuning problem, it is arithmetic: at this size the
account can hold **one or two positions**, traded rarely.

## What it does now

The committed book — chosen by the search, recorded in the seal — is **long
only, one position, reviewed every 60 sessions**. `/pnl` shows the decision in
a sentence and a dated log it writes from its own ledger:

```
IT WANTS TO BUY
INVE-B.ST
INVE-B.ST (56.6% up) — decided from the close of 2026-09-25.
It fills at the next open.
```

Three kinds of line, appended and never rewritten: `plan` (what it wants,
written before the price exists), `fill` (what it paid, at the first open
after), `exit` (what it sold for, on the session the plan named).

**It is deaf between reviews on purpose.** New signals arrive daily and are
ignored, because every change of mind is a round trip. At the review it keeps
the name (costing nothing), switches, or goes to cash.

### It notices when its reasons stop being true, and holds anyway

A buy is made on a pattern — the features that pushed the probability up. Those
are recorded, and the same name is re-read daily: `intact`, `drifting`, or
`broken`. It says so on the page and **never trades on it**, which is measured
rather than assumed. Paired on identical picks and entries, only the exit rule
differing:

| rule | ends with | trades | fees |
|---|---|---|---|
| **hold to review** | **$1,028** | 14 | $56 |
| exit when the model turns | $671 | 58 | $178 |
| exit when the pattern breaks | $495 | 122 | $326 |

The pattern broke in **seven of seven** episodes, and holding on from the break
returned **+8.65% on average** (t +3.31), better in every one. A warning is
free; acting on it spent two thirds of the account.

## How a strategy earns the right to be believed

**The scarce resource is the test set, not compute.** A model trains in
seconds, so three hundred configurations is an afternoon — and the best of three
hundred answers is mostly the luckiest. So:

- The panel is cut in three. The search sees train and validation; the sealed
  period is truncated out before a model is fitted.
- Every trial is appended to `data/search/trials.jsonl`. **32 so far**, which
  puts the bar at **1.61×** what it would have been on the first look.
- `open` refuses to score anything until a configuration is committed in
  writing, with a reason. Deciding before looking is the difference between a
  test and a search.
- A concentrated book is judged by **permutation**: 2,000 random books with the
  identical trade pattern, count and cost. Fiftieth percentile is no skill.

That machinery has now rejected more of my own ideas than it has passed:

| tried | result |
|---|---|
| longer horizons (5, 10, 20 days) | losses shrink because the edge does |
| absolute target | worse than always guessing "up" |
| long-only, high threshold | negative at every threshold |
| market rotation | markets mean-revert; loses to holding the US |
| top-N concentration, daily | top 5 rebalanced daily **wipes the account out** |
| exiting on a broken pattern | worse in seven of seven episodes |
| **top 1, reviewed every 60 sessions** | **96.2% percentile, committed** |

The committed book made +89.3% before costs over eight sealed rebalances, and
**+71.8% with its single best holding removed**. It replicated on validation at
the 99th percentile. Fourteen decisions is not a track record.

## The contest: one champion, a few shadows, and a rule

A champion that is never challenged never improves. A challenger loop with no
brakes is worse than none at all — run enough candidates and one beats the
champion by luck every night, while the ledger reports triumph. So the loop is
deliberately narrow and deliberately slow.

**Narrow.** Every axis it varies is a *decision rule* — how many names, how
long between reviews, what probability to demand. Model capacity, data volume
and freshness are excluded by name, because they are measured flat (see below)
and varying them would raise the bar for everything else and find nothing.

**Slow.** Five new questions a week, never one the ledger has already paid for,
one per cycle. At 32 trials the bar is already 1.61× the first-look bar.

**Clearing the bar wins nothing.** A challenger that beats 95% of matched
random books and ends above buy-and-hold is *registered as a shadow book*: it
starts its own forward record, with its own $500, the same commission by
market, the same review rules. At most three run at once.

**Only the forward record can promote it**, and the record is written daily.
A book reviewed every sixty sessions closes four decisions a year, so a contest
judged on decisions needs two years to say anything — while every session is
already out-of-sample for every book. So each book is **marked every session**,
and challengers are judged on the *paired daily difference* against the funded
book: both see the same market on the same days, so what they share cancels and
what is left is the rule.

The rule, written before any contest ran:

- **120 shared sessions** of record, about six months;
- the challenger **5 points ahead** in total;
- a **paired t of 2** on the daily difference — a far stronger test than
  comparing two noisy totals, and the thing that rejects one lucky day;
- and **no promotion in 180 days**.

A backtest cannot promote anything: clearing one is what got the challenger a
shadow in the first place, and promoting on the same evidence would be
promoting on the thing that has fooled this project three times.

**The guard can only stop.** Separately from promotion, the funded book is
checked every cycle: past a **−20% drawdown** it stands down immediately, and
after six closed decisions it stands down if it was funded on a backtest
expecting to make money and has lost money instead. Standing down means nothing
is funded — and every book keeps recording, because you want to know whether
the one you stopped would have recovered. Nothing in that path can start
trading something.

## What it noticed today

Everything above is built for a verdict months or years away. `briefing.py` is
the output that is useful on the day: a short dated note about what changed,
appended and never edited, so a week of them is a record of what the system
believed at the time.

It only says what it measured — the book's plan, a case that has broken, a
challenger's standing, a name the archive has suddenly filled up with — and
every line carries the number behind it. There is no language model in it and
nothing is generated. Silence is the common outcome and a deliberate one: a
note every morning saying nothing teaches you to stop reading them.

The news spike line is worth singling out, because it is the first thing the
archive has been good for. The block itself is a year from being a feature, but
"this name is in eight articles today against two a day lately — four times its
own normal" needs no history at all, and the collector is already storing the
items.

## Why the model itself is not what improves

Measured on the validation period, one seed spread being 0.47 points:

| question | answer |
|---|---|
| Does more data help? | No. 85k → 342k rows moves accuracy 0.2 pts. |
| Does more capacity help? | No. **Logistic regression matches the network.** |
| Does retraining help? | No. An 18-month-old model matches a refitted one. |
| Do specialists beat the generalist? | No. Worse in six of seven markets. |

The 46 features are the ceiling, and they are nearly all derived from price.
Everything that improved this year was the **decision layer** — turnover, real
commission, hold rules — not the model.

So the scheduler's job is no longer training. It is **accumulating the two
things that cannot be bought later**: the forward paper record, and the news
archive.

## Reading the archive before it is a feature

The block needs a year. That is not a reason to leave two thousand stored items
unread, so `attention.py` reads them under one rule: **it says what it measured,
and it refuses the comparison when the archive cannot support it.**

Two filters do most of the work, and both were written after looking at what the
first version produced:

- **A shared root ticker is one company.** The first run suggested four names,
  every one of them already tracked under another listing -- GSK against GSK.L,
  FER against FER.MC.
- **A name seen in only one company's coverage is probably that company.**
  Cross-listings that do not share a root, like MRSH beside MMC, appear in
  exactly one firm's stories. Turning up in two or more separate names' coverage
  is what promotes a mention to a suggestion.

And one refusal. Counted by capture time the whole archive happened on three
days, because that is when the collector ran -- so every symbol it reaches for
the first time reads as a tenfold spike, ten items being the provider's page
size. **Features count captures; readings count publications**, and nothing that
reads publications can reach the model.

## The model that reads it

The reading layer takes two backends, tried in order: **a model running on
this machine**, then a hosted one. Neither is required -- with no local server
and no API key the pages stay measured, which is the substance anyway.

```bash
ollama serve                          # anything 7-9B at Q4 fits an 8GB card
ollama pull qwen2.5:7b-instruct
```

Local first because it is free, private, and well inside what an 8B model does
reliably: every fact is computed first and handed over in the prompt, so its
whole job is to restate measured findings in a paragraph. It is never given
the raw store to summarise, temperature is 0.2, and it is asked to quote the
number beside each claim so a reader can check it against the figures above.

Override with `TRADER_LLM_LOCAL` (default `http://127.0.0.1:11434`) and
`TRADER_LLM_LOCAL_MODEL`.

## The desk

`/chat` is a transcript, not a chat session. Two kinds of line go into one
append-only file in the order they happened: what it said **unprompted**, when
a scheduler cycle found the archive had moved, and what it said **because you
asked**. Nothing is edited, so what it believed last Tuesday can be held
against what the stores held at the time.

It is silent when nothing changed, and silent entirely when no model is
running. A feed that speaks every cycle teaches you to stop reading it, so the
archive merely gaining items is context rather than news.

**The arithmetic is not the model's job.** `chat.changed` diffs two snapshots
and hands over finished sentences -- "VOLV-B.ST: 3 articles now, down from 6 at
the last line" -- and the model is told to restate them and never to recompute.

That function exists because a 7B asked to compare two numbers it had read in
different places wrote *"1526 articles, up from 1527"*: a fall reported as a
rise. A feed whose whole job is to say what changed cannot be wrong about the
direction of a change, so the comparison happens in code that has tests on it
and the model is left with the writing.

## The news archive

The block is inert until it has a year of history, and `news.readiness`
enforces that rather than handing the model zeros for a decade and real numbers
for last week. **906 items across 85 symbols, 19 days of 365.**

It stores whole items — title, summary, source, publication time, and when
*this machine* saw it. That distinction is the point: **features can be
recomputed from your own archive later; the archive cannot be backfilled.** An
archive fetched next year has been re-ranked by what turned out to matter, and
its timestamps are frequently ingestion times. The scheduler now asks for a
rotating slice of 40 symbols every cycle.

## The gate

`/analysis` gives one call per symbol, and every call is gated on the model
having earned an opinion. Six hurdles: enough distinct days; enough *effective*
rows (ten symbols on one day are not ten verdicts); the model varies; it beats
the training-period majority by more than chance; the edge exceeds the seed
spread and holds across walk-forward windows; and **the money is
distinguishable from luck over the window an order can reach**.

The last one exists because the others were not enough — the logistic control
clears every accuracy test and loses 3.2% a year.

Every hurdle is recorded whether it passed or not. The gate has never opened.

## The three pages

```bash
python run.py                         # http://127.0.0.1:8600
```

**Today** leads with the gate, because the gate has never been open and that is
the thing a reader is least likely to go looking for. Underneath it: all eight
hurdles with the number behind each, then what the paper book intends and the
three features that moved the answer, in sentences.

**Evidence** is the case against trusting it, in four charts. The first is the
one the project earns: the same positions held close-to-close and held from the
next open, drawn from the run's own stored series. They start together at 1.0
and end 9 points apart. The others put the edge beside its own noise floor,
show accuracy climbing with confidence while the tradeable return does not, and
give every walk-forward window both ways.

**News** reads the archive that the model is not allowed to touch for another
year. It groups coverage by market, joins the loudest names to what their price
did, and pulls the explicit company mentions out of the headlines -- roughly a
quarter of them carry one, because the wires write "(LSE:AAL)". Mentions that
resolve to something untracked become suggestions; exchanges that resolve to a
market with no tracked names at all become a larger one.

**Record** is what cannot be re-run: the forward book, the contest, and all 32
trials in the order they were asked, with the committed one marked.

The charts are hand-drawn SVG. Four charts do not pay for a charting
dependency, and each of these is making a specific argument that a generic
chart would soften.

## Statistics that had to be fixed to be honest

- **Overlapping windows.** At a five-day horizon consecutive rows share four
  days of the same move. Skill-free models cleared t = 2 in **39% of runs at
  h=5 and 53% at h=20**; with a Hansen-Hodrick correction, 3% and 8%.
- **The edge's standard error.** The baseline is measured on the same rows, and
  on an absolute target it is shared by every name on a date. Skill-free models
  passed the accuracy hurdle **19% of the time**; clustered by date, 2.7%.
- **The percentile itself.** Measured on 200 draws it rejected a book at 94%
  that is 96.2% on 2,000 — a real decision lost to Monte Carlo noise. Now 2,000.
- **Look-ahead in the neutral band.** It ranked names by their *realised*
  future return and dropped the middle ones from the test rows too. The band now
  thins training only.

## How the pieces fit

```
universe.py   which symbols, and why width is the point
prices.py     daily OHLCV, cached, checked against the period asked for, and
   |          not re-asked of a provider that just answered
features.py   18 per-symbol indicators, every one computable at that close
cross.py      9 -- where a name sits among its peers, lagged one session
macro.py      up to 16 -- the state of the world, from unrevised market data
events.py     3 -- distance to the next announcement
news.py       a point-in-time archive; inert until it has a year
labels.py     the only forward-looking lines; both windows of a day
dataset.py    one cut date, carried not re-derived; purge; scaler on train only
   |
baseline.py   majority, logistic, MLP, walk-forward, noise floor
trainer.py    fit the network, and pack it so the loader can read it back
evaluate.py   accuracy against a training-period baseline, turnover costs, and
   |          every statistic corrected for overlap and clustering
explain.py    the six-hurdle gate, and per-day attribution underneath it
broker.py     what Nordnet and Avanza charge, minimum and all
book.py       a book walked forward in money, and the permutation percentile
search.py     train / validation / sealed test, a trial ledger, commit-then-open
holding.py    the books: each a forward record, one of them funded
challenge.py  the challenger loop -- budgeted, deduped, decision rules only
promote.py    the paired daily comparison, promotion, and the drift guard
briefing.py   what it noticed today, from the stores, written forward
attention.py  what the archive is loud about, and what it is missing
paper.py      the day-trade ledger that came before holding.py -- read only:
   |          nothing writes to it, because 238 names on 500 pays 96% a trip
auto.py       the scheduler: news, books, contest, briefing
web/          three pages: the verdict, the case against it, the record
```

## Limitations worth knowing

**The universe is survivor-biased.** Liquid large caps with long histories, not
index constituents as of any date. Fixing it needs a paid dataset.

**Fourteen decisions is not a track record.** The committed book's evidence is
eight sealed rebalances and six on validation, in a rising market.

**The seal is spent for the current idea.** Low turnover was found by looking at
the sealed period, so the only honest test left for it is forward paper trading.

**Runs stored before the standard-error fix keep the old bar**, and on an
absolute target that bar is too low.

**Costs past one session are approximate** — turnover is charged day to day on
overlapping positions.

**A macro series can stop.** ^VIX3M went unpublished for two months; a series
more than five sessions behind the others is now dropped with its features
rather than carried forward, and picked up again when it resumes.

## Tests

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```

364 tests. The look-ahead ones test the property rather than the implementation
— features computed on a truncated history must match the full one — and one
deliberately introduces a centred window to prove the property test can fail.

Most of the rest are regression tests for measured mistakes: the pinned cut
date, the RSI warm-up, per-trade costs, the training-period baseline, the
sigmoid-to-softmax conversion, the executable return, the overlap correction,
the clustered standard error, the neutral-band leak, the paper ledger that
stopped growing the day it first paid out, the settlement that filled a
238-name book from ten of them, the endpoint that went on reading a field after
the field was deleted, the fill that debited its commission but not the money
it spent — which marked a $500 book at $991 the next session — and the
scheduler loop that ended its own thread on the first unexpected exception
while the page went on reporting that it was running, and the news spike that
fired for five names because the collector had just met them.
