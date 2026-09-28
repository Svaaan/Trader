// The P&L page: what it decided, what it has done, and the numbers underneath.
//
// This page used to open with four panels of statistics and bury the one thing
// worth knowing. It now says the decision first, in a sentence, then the record
// as a dated log -- both read straight off the ledger, so neither can drift
// from what actually happened. Everything that is measurement rather than
// decision sits behind one fold.

const POLL_MS = 20000;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

const money = (v) => (v === null || v === undefined ? "—" : `$${Number(v).toFixed(2)}`);
const pct = (v, d = 2) => (v === null || v === undefined ? "—" : `${(v * 100).toFixed(d)}%`);

function figure(label, value, hint) {
  const box = el("div", "figure");
  box.appendChild(el("div", "figure-label", label));
  box.appendChild(el("div", "figure-value", value));
  if (hint) box.appendChild(el("div", "figure-hint", hint));
  return box;
}

// --- the decision, in one sentence ------------------------------------------

function decision(book) {
  const panel = el("section", "decision");
  const held = book.position;
  const waiting = book.waiting_to_buy;

  if (held) {
    const mark = held.marks[0] || {};
    panel.appendChild(el("p", "decision-label", "It is holding"));
    panel.appendChild(el("h2", "decision-headline", mark.symbol));

    const move = mark.move === null || mark.move === undefined ? null : mark.move;
    const line = el("p", "decision-line");
    line.appendChild(el("span", null,
      `Bought at ${money(mark.price)} on ${mark.session}, now ${money(mark.last)} `));
    line.appendChild(el("span", move < 0 ? "down" : "up", pct(move)));
    panel.appendChild(line);

    panel.appendChild(el("p", "decision-next",
      held.sessions_left > 0
        ? `Sells on ${held.review_on}, ${held.sessions_left} session${held.sessions_left === 1 ? "" : "s"} away.`
        : "The review session has arrived — it decides at the next open."));

    const watch = held.watch;
    if (watch && watch.status !== "intact") {
      panel.appendChild(el("p", watch.status === "broken" ? "decision-warn" : "decision-soft",
        watch.status === "broken"
          ? `The reason it bought has gone — ${watch.note}. It holds anyway: acting on this was measured and lost.`
          : `Weakening — ${watch.note}.`));
    }
  } else if (waiting) {
    const buys = (waiting.buy || [])
      .map((b) => `${b.symbol} (${pct(b.probability_up, 1)} up)`).join(", ");
    panel.appendChild(el("p", "decision-label", "It wants to buy"));
    panel.appendChild(el("h2", "decision-headline", (waiting.buy || [])[0]?.symbol || "—"));
    panel.appendChild(el("p", "decision-line",
      `${buys} — decided from the close of ${waiting.as_of}.`));
    panel.appendChild(el("p", "decision-next",
      "It fills at the next open. The price does not exist yet, which is what makes this a forward record."));
  } else {
    panel.appendChild(el("p", "decision-label", "It is in cash"));
    panel.appendChild(el("h2", "decision-headline", "Nothing held"));
    panel.appendChild(el("p", "decision-next",
      "It decides again after the next run produces signals."));
  }

  const rule = book.book || {};
  panel.appendChild(el("p", "decision-rule",
    `Long only, top ${rule.top_n}, reviewed every ${rule.rebalance_every} sessions. `
    + `${money(book.equity)} from ${money(book.starting_cash)}, ${book.trades || 0} trades, `
    + `${money(book.fees)} in costs.`));
  return panel;
}

// --- the log ------------------------------------------------------------------

function logPanel(entries) {
  if (!entries || !entries.length) return null;

  const panel = el("section", "panel");
  panel.appendChild(el("h3", null, "What it has done"));

  const list = el("ol", "log");
  entries.forEach((entry) => {
    const item = el("li", `log-line is-${entry.kind}`);
    item.appendChild(el("span", "log-when", entry.when));
    item.appendChild(el("span", "log-text", entry.text));
    list.appendChild(item);
  });
  panel.appendChild(list);
  return panel;
}

// --- everything that is measurement rather than decision ----------------------

function numbers(book, account, body) {
  const fold = el("details", "fold");
  fold.appendChild(el("summary", null, "The numbers underneath"));

  if ((book.closed || []).length) {
    fold.appendChild(el("h4", null, "Closed positions"));
    const table = el("table", "table");
    const head = el("tr");
    ["Bought", "Sold", "Name", "In", "Out", "Move", "Profit"]
      .forEach((h) => head.appendChild(el("th", null, h)));
    table.appendChild(head);
    book.closed.slice().reverse().forEach((trade) => {
      (trade.sold || []).forEach((sale) => {
        const row = el("tr");
        [trade.opened, trade.session, sale.symbol, money(sale.bought_at),
         money(sale.price)].forEach((v) => row.appendChild(el("td", null, v)));
        row.appendChild(el("td", sale.gross_return < 0 ? "down" : "up",
          pct(sale.gross_return)));
        row.appendChild(el("td", trade.profit < 0 ? "down" : "up", money(trade.profit)));
        table.appendChild(row);
      });
    });
    fold.appendChild(table);
  }

  if (account && account.days) {
    fold.appendChild(el("h4", null, "The earlier day-trade record"));
    fold.appendChild(el("p", "note",
      `${account.days} settled day(s) of the book that traded every session — the `
      + `shape this account size cannot carry. Kept because it happened: `
      + `${money(account.cash)} from ${money(account.starting_cash)}, hit rate `
      + `${pct(account.hit_rate)}, ${account.pending} still pending.`));

    const brokers = account.brokers || [];
    if (brokers.length) {
      const table = el("table", "table");
      const head = el("tr");
      ["Where", "Trades", "Per trade", "Commission", "Would have ended with"]
        .forEach((h) => head.appendChild(el("th", null, h)));
      table.appendChild(head);
      brokers.forEach((broker) => {
        const row = el("tr");
        row.appendChild(el("td", null, broker.schedule));
        row.appendChild(el("td", null, String(broker.trades || 0)));
        row.appendChild(el("td", null, money(broker.commission_per_trade)));
        row.appendChild(el("td", null, money(broker.commission_paid)));
        row.appendChild(el("td", broker.profit < 0 ? "down" : "up",
          `${money(broker.cash)} (${money(broker.profit)})`));
        table.appendChild(row);
      });
      fold.appendChild(table);
    }
  }

  const divergence = body && body.divergence;
  if (divergence && divergence.note) {
    fold.appendChild(el("h4", null, "Against what the backtest predicted"));
    fold.appendChild(el("p", "note", divergence.note));
  }
  return fold;
}

async function refresh() {
  const host = document.getElementById("pnl");

  try {
    const [pnlResponse, bookResponse] = await Promise.all([
      fetch("/api/pnl").catch(() => null),
      fetch("/api/book").catch(() => null),
    ]);

    const body = pnlResponse && pnlResponse.ok ? await pnlResponse.json() : {};
    const account = body.account || {};
    const book = bookResponse && bookResponse.ok ? await bookResponse.json() : null;

    host.replaceChildren();

    if (!book || book.error) {
      host.appendChild(el("p", "empty",
        "Nothing recorded yet. Train a model and it decides the same evening."));
      return;
    }

    host.appendChild(decision(book));
    const log = logPanel(book.log);
    if (log) host.appendChild(log);
    host.appendChild(numbers(book, account, body));

  } catch (error) {
    host.replaceChildren(el("p", "error", `Could not load: ${error.message}`));
  }
}

document.getElementById("settle").addEventListener("click", async (event) => {
  event.target.disabled = true;
  event.target.textContent = "Working…";
  try {
    await Promise.all([
      fetch("/api/book/advance", { method: "POST" }),
      fetch("/api/pnl/settle", { method: "POST" }),
    ]);
    setTimeout(refresh, 3000);
  } finally {
    setTimeout(() => {
      event.target.disabled = false;
      event.target.textContent = "Check for a decision";
    }, 3000);
  }
});

refresh();
setInterval(refresh, POLL_MS);
