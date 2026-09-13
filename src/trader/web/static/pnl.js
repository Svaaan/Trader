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

async function refresh() {
  const host = document.getElementById("pnl");

  try {
    const response = await fetch("/api/pnl");
    if (!response.ok) throw new Error(`server returned ${response.status}`);
    const body = await response.json();
    const account = body.account || {};

    host.replaceChildren();

    if (!account.days && !account.pending) {
      host.appendChild(el("p", "empty",
        "Nothing recorded yet. Train a model and the ledger starts the same "
        + "evening — the first line settles at the next open."));
      return;
    }

    host.appendChild(el("p", "analysis-meta",
      `Filled at the open after each signal, closed at that session's close, `
      + `${pct(body.cost_per_side, 2)} charged each way.`));

    [standingPanel(account), divergencePanel(body), curvePanel(account)]
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
