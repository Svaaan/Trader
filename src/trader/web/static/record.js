// The record: what it has actually done, and what it asked before doing it.
//
// This page answers the one question the other two cannot. The front page says
// what the model thinks; the evidence page says how it scored on history it was
// held out from. Both of those are measurements on the past, and the past can
// be mined. This is the part that cannot: a decision written down before the
// price existed, and a ledger of every configuration that was asked about,
// appended in the order it was asked.
//
// So the trial ledger is shown in full rather than ranked. The trials that
// failed are what make the one that passed mean anything -- thirty-two
// questions is why the bar the thirty-third has to clear is higher.

import {
  el, pct, pp, num, plain, money, count, signedPct, sign, getJSON, poll,
  stat, statRow, sectionHead, table, tag,
} from "ui";

const POLL_MS = 30000;

// --- the funded book ----------------------------------------------------------

function bookSection(book) {
  if (!book || book.error) return null;

  const section = el("section");
  section.appendChild(sectionHead(
    "the funded book",
    book.book_id || "champion",
    book.why
      ? `The funded book is ${book.why}. Every line below was written before `
        + `the price it refers to existed.`
      : "Every line below was written before the price it refers to existed."));

  section.appendChild(statRow(
    stat("equity", money(book.equity), `from ${money(book.starting_cash)}`,
      book.profit === 0 ? null : sign(book.profit) === "up" ? "good" : "bad"),
    stat("return", signedPct(book.return), "since it was funded",
      book.return === 0 ? null : book.return > 0 ? "good" : "bad"),
    stat("closed", count(book.realised || 0), "decisions with a result"),
    stat("costs paid", money(book.fees), book.schedule || null),
    stat("trades", count(book.trades || 0), "fills, both directions"),
  ));

  const log = book.log || [];
  if (log.length) {
    const panel = el("div", "panel");
    panel.appendChild(el("h3", null, "What it has done"));
    const list = el("ul", "ledger");
    log.forEach((entry) => {
      const item = el("li", `is-${entry.kind}`);
      item.appendChild(el("span", "ledger-when", entry.when));
      item.appendChild(el("span", "ledger-text", entry.text));
      list.appendChild(item);
    });
    panel.appendChild(list);
    section.appendChild(panel);
  }

  const closed = book.closed || [];
  if (closed.length) {
    const fold = el("details", "fold");
    fold.appendChild(el("summary", null, `Closed positions (${closed.length})`));
    const rows = [];
    closed.slice().reverse().forEach((trade) => {
      (trade.sold || []).forEach((sale) => {
        rows.push({
          cells: [
            sale.symbol, trade.opened, trade.session,
            money(sale.bought_at), money(sale.price),
            { text: signedPct(sale.gross_return), className: sign(sale.gross_return) },
            { text: money(trade.profit), className: sign(trade.profit) },
          ],
        });
      });
    });
    fold.appendChild(table(
      ["Name", "Bought", "Sold", "In", "Out", "Move", "Profit"], rows));
    section.appendChild(fold);
  }

  return section;
}

// --- the contest ---------------------------------------------------------------

function contestSection(contest) {
  if (!contest || contest.error) return null;
  const books = contest.books || [];

  const section = el("section");
  section.appendChild(sectionHead(
    "the contest",
    books.length > 1
      ? `One funded book, ${books.length - 1} shadow${books.length === 2 ? "" : "s"}`
      : "One funded book, nothing challenging it yet",
    "A challenger that clears the bar on validation does not get funded — "
    + "it starts a forward record of its own. Only that record can promote it, "
    + "and only after a hundred and twenty shared sessions and a paired t of "
    + "two on the daily difference. A backtest cannot promote anything."));

  if (contest.guard && contest.guard.stand_down) {
    section.appendChild(el("p", "warn-line",
      `Trading is stopped — ${contest.guard.why}. The records keep running, `
      + `so you can see whether it would have recovered.`));
  }

  section.appendChild(table(
    ["Book", "Rule", "Closed", "Equity", "Return", "Status"],
    books.map((entry) => {
      const rule = entry.rule || entry.book || {};
      return {
        className: entry.funded ? "is-subject" : null,
        cells: [
          entry.book_id,
          `top ${rule.top_n}, every ${rule.rebalance_every}`,
          count(entry.realised || 0),
          money(entry.equity),
          { text: signedPct(entry.return), className: sign(entry.profit) },
          tag(entry.funded ? "funded" : "shadow", entry.funded ? "forward" : "quiet"),
        ],
      };
    }),
  ));

  const promotion = contest.promotion || {};
  const spare = Math.max((contest.budget || 0) - (contest.asked || 0), 0);

  section.appendChild(statRow(
    stat("promotion", promotion.promote ? "ready" : "not yet",
      promotion.why || null),
    stat("questions this week", count(contest.asked),
      spare
        ? `${spare} of ${contest.budget} left in the budget`
        : `the budget of ${contest.budget} is spent until the week turns`,
      spare ? null : "bad"),
    stat("rules untried", count(contest.waiting),
      "the challenger loop works through these"),
    stat("drift guard", contest.guard?.stand_down ? "stood down" : "watching",
      contest.guard?.why || null),
  ));

  (promotion.challengers || []).forEach((challenger) => {
    const paired = challenger.paired || {};
    section.appendChild(el("p", "note",
      `${challenger.book_id}: ${signedPct(paired.ahead)} against the funded book `
      + `over ${paired.sessions || 0} shared sessions, paired t ${num(paired.t)}.`));
  });

  return section;
}

// --- the trial ledger ----------------------------------------------------------

function trialSection(body) {
  const trials = (body || {}).trials || [];
  if (!trials.length) return null;

  const seal = body.seal || {};
  const committed = seal.committed || null;

  const section = el("section");
  section.appendChild(sectionHead(
    "the search ledger",
    `${body.total} trials, appended in the order they were asked`,
    "Every configuration is written down before it is scored, and the file is "
    + "append-only. That is what makes the count usable: asking thirty-two "
    + "questions of the same data raises the bar the next answer has to clear, "
    + "and a ledger you can rewrite cannot be used to correct anything."));

  section.appendChild(statRow(
    stat("trials asked", count(body.total), "on validation only"),
    stat("test set opened", count((seal.opened || []).length),
      "each opening is spent and recorded"),
    stat("committed", committed ? "yes" : "no",
      committed ? "the rule the funded book follows" : "nothing committed"),
  ));

  if (committed) {
    const panel = el("div", "panel");
    panel.appendChild(el("h3", null, "What was committed, and why"));
    panel.appendChild(el("p", "note", committed.why || ""));
    section.appendChild(panel);
  }

  // Two kinds of question live in this ledger and they are not scored the same
  // way. The early ones ask whether an edge survives: they carry an accuracy
  // and a comparable edge. The later ones ask whether a book makes money at
  // $500 with real commission: they carry an ending balance and a percentile
  // against matched random books, and their `edge` is a placeholder zero.
  // Printing one set of columns for both was showing "+0.00pp" as though the
  // money trials had measured an edge of nothing.
  section.appendChild(table(
    ["#", "Rule", "Sharpe", "t", "Ended with", "Vs random", "Edge"],
    trials.map((trial) => {
      const v = trial.validation || {};
      const money = v.final_equity !== null && v.final_equity !== undefined;
      const sharpe = v.executable_sharpe ?? v.strategy_sharpe;
      const committedHere = committed && sameRule(trial, committed);

      return {
        className: committedHere ? "is-subject" : null,
        cells: [
          trial.trial ?? "—",
          describeBook(trial.book, trial.spec),
          { text: num(sharpe), className: sign(sharpe) },
          num(v.executable_tstat ?? v.strategy_tstat),
          money ? money_(v.final_equity, v.random_median) : "—",
          money ? percentile(v.random_percentile) : "—",
          v.edge_comparable === false || v.edge === null || v.edge === undefined
            ? "—"
            : { text: pp(v.edge), className: v.edge > 0 ? "up" : "flat" },
        ],
      };
    }),
  ));

  section.appendChild(el("p", "note",
    "Highlighted: the rule that was committed and is now running forward. A "
    + "trial scored in money reports what $500 ended as and where that sat "
    + "against matched random books; a trial scored on the signal reports an "
    + "edge. Neither column is filled in for the other kind, because a "
    + "placeholder zero reads as a measurement."));
  return section;
}

function money_(equity, median) {
  return median ? `${money(equity, 0)} / ${money(median, 0)}` : money(equity, 0);
}

function percentile(value) {
  if (value === null || value === undefined) return "—";
  const at = Math.round(value * 100);
  const teen = at % 100 >= 11 && at % 100 <= 13;
  const suffix = teen ? "th" : { 1: "st", 2: "nd", 3: "rd" }[at % 10] || "th";
  return { text: `${at}${suffix}`, className: at >= 95 ? "up" : "flat" };
}

function describeBook(book, spec) {
  if (!book) return spec ? `${spec.target || "?"} h${spec.horizon ?? "?"}` : "—";
  const parts = [book.long_only ? "long only" : "both ways"];
  if (book.top_n) parts.push(`top ${book.top_n}`);
  if (book.rebalance_every) parts.push(`every ${book.rebalance_every}`);
  if (book.min_probability && book.min_probability !== 0.5) {
    parts.push(`p≥${book.min_probability}`);
  }
  return parts.join(", ");
}

// Same rule, same spec -- which is what "this is the committed one" means.
function sameRule(trial, committed) {
  const a = trial.book || {};
  const b = committed.book || {};
  return ["long_only", "top_n", "rebalance_every", "min_probability"]
    .every((key) => a[key] === b[key]);
}

// The pre-registration notes are paragraphs. The table wants a clause.
function shorten(note, limit = 90) {
  if (!note) return "—";
  const first = String(note).split(/[.;]/)[0].trim();
  return first.length > limit ? `${first.slice(0, limit)}…` : first;
}

// --- the earlier ledger --------------------------------------------------------

function paperSection(account) {
  if (!account || !account.days) return null;

  const fold = el("details", "fold");
  fold.appendChild(el("summary", null,
    `The earlier day-trade record (${account.days} settled day${account.days === 1 ? "" : "s"})`));

  fold.appendChild(el("p", "note",
    `The book that traded every session, kept because it happened and read-only `
    + `because nothing writes to it any more: on a ${money(account.starting_cash)} `
    + `account a 238-name daily book pays about 96% of a position per round trip. `
    + `It ended with ${money(account.cash)}, hit rate ${pct(account.hit_rate)}, `
    + `${account.pending} still pending.`));

  const brokers = account.brokers || [];
  if (brokers.length) {
    fold.appendChild(table(
      ["Where", "Trades", "Per trade", "Commission", "Would have ended with"],
      brokers.map((broker) => ({
        cells: [
          broker.schedule, count(broker.trades || 0),
          money(broker.commission_per_trade), money(broker.commission_paid),
          { text: `${money(broker.cash)} (${money(broker.profit)})`,
            className: sign(broker.profit) },
        ],
      }))));
  }
  return fold;
}

// --- the page ------------------------------------------------------------------

async function refresh() {
  const host = document.getElementById("record");

  const [book, contest, trials, pnl] = await Promise.all([
    getJSON("/api/book"),
    getJSON("/api/contest"),
    getJSON("/api/trials"),
    getJSON("/api/pnl"),
  ]);

  if (!book && !contest && !trials) {
    host.replaceChildren(el("p", "error", "Could not reach the server."));
    return;
  }

  const head = el("section", "verdict");
  head.appendChild(el("div", "verdict-kicker", "forward only"));
  head.appendChild(el("h1", "verdict-word", "The record"));
  head.appendChild(el("p", "verdict-why",
    "A decision written down before the price existed, and every question that "
    + "was asked of the data before it. Neither can be re-run, which is the "
    + "only reason either is worth reading."));

  const parts = [head, bookSection(book), contestSection(contest),
    trialSection(trials), paperSection((pnl || {}).account)];

  const divergence = (pnl || {}).divergence;
  if (divergence && divergence.note) {
    const section = el("section");
    section.appendChild(sectionHead("against the backtest",
      "What the forward record says about the prediction"));
    section.appendChild(el("p", "note", divergence.note));
    parts.push(section);
  }

  host.replaceChildren(...parts.filter(Boolean));
}

document.getElementById("advance")?.addEventListener("click", async (event) => {
  const button = event.currentTarget;
  button.disabled = true;
  button.textContent = "Working…";
  try {
    await fetch("/api/book/advance", { method: "POST" });
  } finally {
    setTimeout(() => {
      button.disabled = false;
      button.textContent = "Check for a decision";
      refresh();
    }, 3000);
  }
});

poll(refresh, POLL_MS);
