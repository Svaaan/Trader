// The forward paper record.
//
// The order here is deliberate and it is not the order a trading app would use.
// A trading app leads with the running total, because that is the number people
// want. This leads with how many days there are, because on a record this short
// the total is the least informative thing on the page -- ten days of coin flips
// produce impressive percentages in both directions, and printing one at the top
// would teach exactly the wrong reflex.
//
// createElement throughout: symbol names come from a data provider.

const POLL_MS = 30000;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

const money = (v) => (v === null || v === undefined ? "—" : `$${Number(v).toFixed(2)}`);
const pct = (v, d = 2) => (v === null || v === undefined ? "—" : `${(v * 100).toFixed(d)}%`);
const num = (v, d = 2) => (v === null || v === undefined ? "—" : Number(v).toFixed(d));

function figure(label, value, hint) {
  const box = el("div", "figure");
  box.appendChild(el("div", "figure-label", label));
  box.appendChild(el("div", "figure-value", value));
  if (hint) box.appendChild(el("div", "figure-hint", hint));
  return box;
}

// --- how much of a record there is -----------------------------------------

function standingPanel(account) {
  const panel = el("section", "panel");
  panel.appendChild(el("h3", null, "How much of a record this is"));

  const grid = el("div", "figures");
  grid.appendChild(figure("Settled days", (account.days || 0).toLocaleString(),
    account.pending ? `${account.pending} waiting on prices` : "nothing pending"));
  grid.appendChild(figure("t", num(account.tstat),
    "under 2 is not distinguishable from luck"));
  grid.appendChild(figure("Account", money(account.cash),
    `from ${money(account.starting_cash)}`));
  grid.appendChild(figure("Profit", money(account.profit),
    pct(account.return)));
  panel.appendChild(grid);

  // The verdict leads, and it is written to refuse a small sample.
  panel.appendChild(el("p", account.days >= 120 ? "verdict" : "verdict is-thin",
    account.verdict || ""));

  if (account.days) {
    const risk = el("div", "figures");
    risk.appendChild(figure("Sharpe", num(account.sharpe),
      account.days < 120 ? "annualised from too few days to mean it"
        : "annualised"));
    risk.appendChild(figure("Winning days",
      `${account.wins} / ${account.days}`,
      pct(account.days ? account.wins / account.days : 0, 0)));
    risk.appendChild(figure("Hit rate", pct(account.hit_rate),
      "of individual positions"));
    risk.appendChild(figure("Worst drawdown", pct(account.max_drawdown),
      "peak to trough"));
    panel.appendChild(risk);
  }

  if (account.shadow_days) {
    panel.appendChild(el("p", "warn",
      `${account.shadow_days} of ${account.days} settled days were recorded `
      + "while the gate was shut — the model had not earned an opinion, so "
      + "these are a record of what it would have done, not of a "
      + "recommendation it was entitled to make."));
  }

  return panel;
}

// --- does live match what the backtest promised? ---------------------------

function divergencePanel(body) {
  const d = body.divergence || {};
  const backtest = body.backtest || {};

  const panel = el("section", "panel");
  panel.appendChild(el("h3", null, "Against what the backtest predicted"));

  if (!d.comparable) {
    panel.appendChild(el("p", "note", d.note || "Not enough to compare yet."));
    return panel;
  }

  const grid = el("div", "figures");
  grid.appendChild(figure("Backtest said", num(d.expected_sharpe),
    "Sharpe, open to close, after costs"));
  grid.appendChild(figure("Live", num(d.actual_sharpe), "Sharpe, same basis"));
  grid.appendChild(figure("Gap", num(d.gap),
    `${num(Math.abs(d.gap) / (d.standard_error || 1), 1)} standard errors`));
  panel.appendChild(grid);

  panel.appendChild(el("p", d.diverged ? "warn" : "note", d.note));

  // This is the only feedback loop the design permits, and saying so on the
  // page is part of keeping it that way.
  panel.appendChild(el("p", "note",
    "This comparison is the only thing the system does with its own results. "
    + "It does not train on them — a model tuned on its paper record turns the "
    + "one un-mineable measurement in the project into another training set. "
    + "Noticing that live has come apart from the backtest is useful; learning "
    + "from it is not."));

  if (!backtest.gate_open) {
    panel.appendChild(el("p", "note",
      "The gate is currently shut, so the backtest figure above is what the "
      + "model would have to beat before any of this counts as a strategy."));
  }

  return panel;
}

// --- the curve --------------------------------------------------------------

function curvePanel(account) {
  const equity = account.equity || [];
  if (!equity.length) return null;

  const panel = el("section", "panel");
  panel.appendChild(el("h3", null, "Every settled day"));

  const values = equity.map((e) => e.equity);
  const low = Math.min(...values, account.starting_cash);
  const high = Math.max(...values, account.starting_cash);
  const span = high - low || 1;

  // A sparkline drawn as divs. A charting library for one line on a page that
  // is mostly warning people not to read too much into the line would be an
  // odd trade.
  const chart = el("div", "sparkline");
  equity.forEach((point) => {
    const bar = el("div", "spark");
    bar.style.height = `${8 + ((point.equity - low) / span) * 92}%`;
    if (point.return < 0) bar.classList.add("is-down");
    bar.title = `${point.session}: ${money(point.equity)} (${pct(point.return)})`;
    chart.appendChild(bar);
  });
  panel.appendChild(chart);

  const table = el("table", "table");
  const head = el("tr");
  ["Session", "Positions", "Return", "Equity", "Hit rate"].forEach((h) =>
    head.appendChild(el("th", null, h)));
  table.appendChild(head);

  // Newest first: the last few days are what somebody actually came to see.
  equity.slice().reverse().slice(0, 30).forEach((point) => {
    const row = el("tr", point.return >= 0 ? "is-ok" : "is-bad");
    row.appendChild(el("td", null, point.session));
    row.appendChild(el("td", null, point.positions));
    row.appendChild(el("td", null, pct(point.return, 3)));
    row.appendChild(el("td", null, money(point.equity)));
    row.appendChild(el("td", null, pct(point.hit_rate, 0)));
    table.appendChild(row);
  });
  panel.appendChild(table);

  if (equity.length > 30) {
    panel.appendChild(el("p", "note",
      `Showing the most recent 30 of ${equity.length} settled days.`));
  }
  return panel;
}

// --- assembly ---------------------------------------------------------------

function bookPanel(book) {
  if (!book || book.error) return null;

  const panel = el("section", "panel");
  panel.appendChild(el("h3", null, "The committed book"));

  const rule = book.book || {};
  panel.appendChild(el("p", "note",
    `Long only, top ${rule.top_n}, reviewed every ${rule.rebalance_every} sessions, `
    + `at ${book.schedule}. It buys at the first open after it decides, and sells `
    + `at the open of the session it said it would — not on the day the number looks good.`));

  const grid = el("div", "figures");
  grid.appendChild(figure("Account", money(book.equity),
    `from ${money(book.starting_cash)}`));
  grid.appendChild(figure("Profit", money(book.profit), pct(book.return)));
  grid.appendChild(figure("Trades", String(book.trades || 0),
    `${money(book.fees)} in fees`));
  panel.appendChild(grid);

  const waiting = book.waiting_to_buy;
  const held = book.position;

  if (held) {
    const mark = held.marks[0] || {};
    const state = el("div", "call");
    state.appendChild(el("h4", null, `Holding ${mark.symbol}`));
    state.appendChild(el("p", null,
      `Bought ${money(mark.price)} on ${mark.session}, now ${money(mark.last)} `
      + `(${pct(mark.move)}). Worth ${money(held.worth)}.`));
    state.appendChild(el("p", "verdict",
      held.sessions_left > 0
        ? `Sells on ${held.review_on} — ${held.sessions_left} session(s) away. `
          + `Held ${held.held_sessions} so far, and nothing is traded in between.`
        : `The review session has arrived: it sells at the next open.`));

    const because = held.bought_because || [];
    if (because.length) {
      state.appendChild(el("p", "note",
        `It bought on: ${because.map((d) => d.feature.replace(/_/g, " ")).join(", ")}.`));
    }

    const watch = held.watch;
    if (watch) {
      const line = el("p", watch.status === "broken" ? "error"
        : watch.status === "drifting" ? "note" : "note");
      line.textContent = watch.status === "broken"
        ? `⚠ The case has broken — ${watch.note}. It is holding anyway: a warning `
          + `is free, and changing its mind costs a round trip it has not earned.`
        : watch.status === "drifting"
          ? `Drifting — ${watch.note}.`
          : `On its pattern — ${watch.note}.`;
      state.appendChild(line);
    }
    if (held.trusted === false) {
      state.appendChild(el("p", "note",
        "Recorded while the gate was shut, so this is a record of what it would "
        + "have done, not a recommendation."));
    }
    panel.appendChild(state);
  } else if (waiting) {
    const buys = (waiting.buy || []).map((b) =>
      `${b.symbol} (${pct(b.probability_up)} up)`).join(", ");
    const state = el("div", "call");
    state.appendChild(el("h4", null, `Wants to buy ${buys}`));
    state.appendChild(el("p", null,
      `Decided from the close of ${waiting.as_of}. It fills at the next open — `
      + `the price does not exist yet, which is what makes this a forward record.`));
    panel.appendChild(state);
  } else {
    panel.appendChild(el("p", "empty",
      "In cash. It plans a buy after the next run produces signals."));
  }

  const closed = book.closed || [];
  if (closed.length) {
    panel.appendChild(el("h4", null, "What it has done"));
    const table = el("table", "table");
    const head = el("tr");
    ["Bought", "Sold", "Name", "In", "Out", "Move", "Profit"]
      .forEach((h) => head.appendChild(el("th", null, h)));
    table.appendChild(head);
    closed.slice().reverse().forEach((trade) => {
      (trade.sold || []).forEach((sale) => {
        const row = el("tr");
        row.appendChild(el("td", null, trade.opened));
        row.appendChild(el("td", null, trade.session));
        row.appendChild(el("td", null, sale.symbol));
        row.appendChild(el("td", null, money(sale.bought_at)));
        row.appendChild(el("td", null, money(sale.price)));
        row.appendChild(el("td", sale.gross_return < 0 ? "is-down" : "is-up",
          pct(sale.gross_return)));
        row.appendChild(el("td", trade.profit < 0 ? "is-down" : "is-up",
          money(trade.profit)));
        table.appendChild(row);
      });
    });
    panel.appendChild(table);
  }
  return panel;
}

function brokerPanel(account) {
  const brokers = account.brokers || [];
  if (!brokers.length) return null;

  const panel = el("section", "panel");
  panel.appendChild(el("h3", null, "What a real broker would have taken"));

  const table = el("table", "table");
  const head = el("tr");
  ["Where", "Trades", "Per trade", "Commission", "Ends with", "Minimum binds below"]
    .forEach((h) => head.appendChild(el("th", null, h)));
  table.appendChild(head);

  brokers.forEach((broker) => {
    const row = el("tr");
    row.appendChild(el("td", null, broker.schedule));
    row.appendChild(el("td", null, (broker.trades || 0).toLocaleString()));
    row.appendChild(el("td", null, money(broker.commission_per_trade)));
    row.appendChild(el("td", null, money(broker.commission_paid)));
    const ends = el("td", broker.profit < 0 ? "is-down" : "is-up",
      `${money(broker.cash)} (${money(broker.profit)})`);
    row.appendChild(ends);
    row.appendChild(el("td", null,
      Number.isFinite(broker.break_even_value) ? money(broker.break_even_value) : "—"));
    table.appendChild(row);
  });
  panel.appendChild(table);

  const real = brokers.find((b) => b.source && !b.source.startsWith("evaluate"));
  if (real) {
    panel.appendChild(el("p", "note",
      `A percentage or a minimum, whichever is larger — so below `
      + `${money(real.break_even_value)} a position, every trade costs the same and `
      + `halving the position doubles the cost as a share of it. This book holds `
      + `every name it has an opinion on, which makes each position small. `
      + `Read off ${real.source} on ${real.checked}; they change, and your class `
      + `and tier may differ.`));
  }
  return panel;
}

async function refresh() {
  const host = document.getElementById("pnl");

  try {
    const [response, bookResponse] = await Promise.all([
      fetch("/api/pnl"),
      fetch("/api/book").catch(() => null),
    ]);
    if (!response.ok) throw new Error(`server returned ${response.status}`);
    const body = await response.json();
    const account = body.account || {};
    const book = bookResponse && bookResponse.ok ? await bookResponse.json() : null;

    host.replaceChildren();

    if (!account.days && !account.pending && !(book && (book.position || book.waiting_to_buy))) {
      host.appendChild(el("p", "empty",
        "Nothing recorded yet. Train a model and the ledger starts the same "
        + "evening — the first line settles at the next open."));
      return;
    }

    host.appendChild(el("p", "analysis-meta",
      `Filled at the open after each signal, closed at that session's close, `
      + `${pct(body.cost_per_side, 2)} charged each way.`));

    [bookPanel(book), standingPanel(account), brokerPanel(account),
     divergencePanel(body), curvePanel(account)]
      .filter(Boolean)
      .forEach((panel) => host.appendChild(panel));

  } catch (error) {
    host.replaceChildren(el("p", "error", `Could not load: ${error.message}`));
  }
}

document.getElementById("settle").addEventListener("click", async (event) => {
  event.target.disabled = true;
  event.target.textContent = "Settling…";
  try {
    await fetch("/api/pnl/settle", { method: "POST" });
    // The fills happen in the background; give them a moment before redrawing.
    setTimeout(refresh, 3000);
  } finally {
    setTimeout(() => {
      event.target.disabled = false;
      event.target.textContent = "Settle what has happened";
    }, 3000);
  }
});

refresh();
setInterval(refresh, POLL_MS);
