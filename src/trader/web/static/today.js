// The front page: the verdict, the gate, and today's decision.
//
// Ordered by what a reader most needs and is least likely to look for. The
// model is accurate and loses money, so the first thing on the page is that
// the gate is shut and the sentence explaining why -- before the accuracy,
// before the decision, before anything that could be mistaken for advice.
//
// The decision comes second and is framed as what the paper book intends, not
// as a recommendation. It is shown at all because a forward record that cannot
// be inspected is not a record anybody should trust either.

import {
  el, pct, pp, num, money, count, getJSON, poll, pips, gateRail, stat, statRow,
  sectionHead, tag, signedPct, hurdleName,
} from "ui";

const POLL_MS = 20000;

// --- the verdict --------------------------------------------------------------

function verdict(run) {
  const trust = run.trust || {};
  const head = run.headline || {};
  const open = Boolean(trust.trusted);

  const panel = el("section", `verdict ${open ? "is-open" : "is-shut"}`);
  panel.appendChild(el("div", "verdict-kicker",
    `Model ${run.run_id} · graded on ${head.test_from} to ${head.test_to}`));
  panel.appendChild(el("h1", "verdict-word",
    open ? "The gate is open" : "The gate is shut"));
  panel.appendChild(el("p", "verdict-why", trust.reason || run.verdict || ""));

  const strip = el("div", "verdict-strip");
  const add = (label, value, tone) => {
    const box = el("div", "verdict-stat");
    box.appendChild(el("b", tone ? `is-${tone}` : null, value));
    box.appendChild(el("span", null, label));
    strip.appendChild(box);
  };

  add("accuracy", pct(head.accuracy), "good");
  add(`over a ${pct(head.baseline_accuracy)} baseline`, pp(head.edge), "good");
  add("t of the return an order can reach", num(head.executable_tstat), "bad");
  add("sessions graded", count(head.days));
  panel.appendChild(strip);

  return panel;
}

// --- the gate -----------------------------------------------------------------

function gate(run) {
  const checks = (run.trust || {}).checks || [];
  if (!checks.length) return null;

  const failed = checks.filter((c) => !c.passed);
  const section = el("section");
  section.appendChild(sectionHead(
    "the gate",
    failed.length === 1
      ? `${checks.length - 1} of ${checks.length} cleared. The one it fails asks `
        + `whether ${hurdleName(failed[0].name).toLowerCase()}.`
      : failed.length
        ? `${failed.length} of ${checks.length} hurdles are shut.`
        : `All ${checks.length} hurdles cleared.`,
    "Every hurdle is recorded whether it passed or not, and the last one exists "
    + "because the others were not enough: the logistic control clears every "
    + "accuracy test and still loses money."));

  const bar = el("div", "panel");
  bar.appendChild(pips(checks));
  section.appendChild(bar);
  section.appendChild(gateRail(checks));
  return section;
}

// --- what it intends to do ----------------------------------------------------

function decision(book, signals) {
  if (!book || book.error) return null;

  const section = el("section");
  section.appendChild(sectionHead(
    "the forward record",
    "What the paper book intends",
    "Decided from a close, filled at an open that had not happened yet. No "
    + "order is placed and no broker credential exists in this project — "
    + "the record is kept so that the decision can be checked later against a "
    + "price nobody could have known when it was made."));

  const held = book.position;
  const waiting = book.waiting_to_buy;
  const panel = el("div", `decision${held || waiting ? "" : " is-cash"}`);

  if (waiting && !held) {
    const buy = (waiting.buy || [])[0] || {};
    panel.appendChild(el("p", "decision-label", "Wants to buy at the next open"));
    panel.appendChild(el("h2", "decision-name", buy.symbol || "—"));
    const line = el("p", "decision-line");
    line.appendChild(el("span", null, "It puts "));
    line.appendChild(el("strong", null, pct(buy.probability_up, 1)));
    line.appendChild(el("span", null,
      ` on this name rising against the panel, decided from the close of ${waiting.as_of}.`));
    panel.appendChild(line);
    panel.appendChild(reasons(signals[buy.symbol]));
  } else if (held) {
    const mark = (held.marks || [])[0] || {};
    panel.appendChild(el("p", "decision-label", "Holding"));
    panel.appendChild(el("h2", "decision-name", mark.symbol || "—"));
    const line = el("p", "decision-line");
    line.appendChild(el("span", null, `Bought at ${money(mark.price)} on ${mark.session}, now `));
    line.appendChild(el("strong", null, money(mark.last)));
    line.appendChild(el("span", `  ${mark.move < 0 ? "down" : "up"}`,
      `  ${signedPct(mark.move)}`));
    panel.appendChild(line);

    panel.appendChild(el("p", "decision-line",
      held.sessions_left > 0
        ? `Reviewed on ${held.review_on}, ${held.sessions_left} session`
          + `${held.sessions_left === 1 ? "" : "s"} away.`
        : "The review session has arrived — it decides at the next open."));

    const watch = held.watch;
    if (watch && watch.status === "broken") {
      panel.appendChild(el("p", "warn-line",
        `The reason it bought has gone — ${watch.note}. It holds anyway: `
        + `acting on a broken case was measured and lost in seven of seven episodes.`));
    } else if (watch && watch.status === "drifting") {
      panel.appendChild(el("p", "decision-line", `Weakening — ${watch.note}.`));
    }
    panel.appendChild(reasons(signals[mark.symbol]));
  } else {
    panel.appendChild(el("p", "decision-label", "In cash"));
    panel.appendChild(el("h2", "decision-name", "Nothing held"));
    panel.appendChild(el("p", "decision-line",
      "It decides again after the next run produces signals."));
  }

  section.appendChild(panel);

  const rule = book.rule || book.book || {};
  section.appendChild(statRow(
    stat("rule", `top ${rule.top_n}`,
      `long only, reviewed every ${rule.rebalance_every} sessions`),
    stat("equity", money(book.equity), `from ${money(book.starting_cash)}`),
    stat("costs paid", money(book.fees),
      book.schedule ? `at ${book.schedule} rates` : null),
    stat("closed decisions", count(book.realised || 0),
      "each one a sealed forward result"),
  ));

  return section;
}

// The three features that moved the answer most, in the model's own words.
function reasons(signal) {
  const list = el("ul", "reasons");
  const rows = (signal || {}).reasons || [];
  if (!rows.length) {
    list.appendChild(el("li", "reason-text",
      "No per-feature reasons were recorded for this name."));
    return list;
  }
  rows.forEach((reason) => {
    const item = el("li", "reason");
    item.appendChild(el("span", "reason-weight", pp(reason.effect)));
    item.appendChild(el("span", "reason-text", reason.sentence));
    list.appendChild(item);
  });
  return list;
}

// --- what it noticed ----------------------------------------------------------

function briefing(notes) {
  const note = (notes || [])[0];
  if (!note || !(note.lines || []).length) return null;

  const section = el("section");
  section.appendChild(sectionHead(
    "the briefing",
    `What it noticed on ${note.date}`,
    "Computed from the stores and written forward, once a day. A quiet day "
    + "writes nothing, because a note every morning saying nothing teaches you "
    + "to stop reading them."));

  const panel = el("div", "panel");
  const list = el("ul", "ledger");
  note.lines.forEach((line) => {
    const item = el("li", `is-${line.kind}`);
    item.appendChild(el("span", "ledger-when", line.kind));
    item.appendChild(el("span", "ledger-text", line.text));
    list.appendChild(item);
  });
  panel.appendChild(list);
  section.appendChild(panel);
  return section;
}

// --- the page -----------------------------------------------------------------

async function refresh() {
  const host = document.getElementById("today");
  const body = await getJSON("/api/decision");

  if (!body) {
    host.replaceChildren(el("p", "error", "Could not reach the server."));
    return;
  }

  const parts = [];
  if (body.run) {
    parts.push(verdict(body.run), gate(body.run));
  } else {
    const empty = el("section", "verdict is-shut");
    empty.appendChild(el("div", "verdict-kicker", "no model"));
    empty.appendChild(el("h1", "verdict-word", "Nothing graded yet"));
    empty.appendChild(el("p", "verdict-why",
      body.why || "Train a model and it is graded on rows it never saw."));
    parts.push(empty);
  }

  parts.push(decision(body.book, body.signals || {}));
  parts.push(briefing(body.briefing));
  host.replaceChildren(...parts.filter(Boolean));
}

// --- the controls in the header ----------------------------------------------

function describeAuto(state) {
  if (!state) return "";
  if (!state.running) {
    return state.stopped_because
      ? `Scheduler off — ${state.stopped_because}`
      : "Scheduler off";
  }
  const last = state.last || {};
  const did = last.error ? `failed: ${last.error}`
    : last.trained ? `trained ${last.trained}`
      : last.skipped || "starting";
  return `Scheduler on, every ${state.interval_minutes} min · `
    + `${state.cycles || 0} cycle(s) · ${did}`;
}

async function refreshAuto() {
  const state = await getJSON("/api/auto");
  const line = document.getElementById("autoState");
  const button = document.getElementById("autoTrain");
  if (line) line.textContent = describeAuto(state);
  if (button && state) {
    button.dataset.running = state.running ? "yes" : "no";
    button.textContent = state.running ? "Stop the scheduler" : "Start the scheduler";
  }
}

document.getElementById("autoTrain")?.addEventListener("click", async (event) => {
  const button = event.currentTarget;
  const running = button.dataset.running === "yes";
  button.disabled = true;
  try {
    await fetch(running ? "/api/auto/stop" : "/api/auto/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: running ? null : JSON.stringify({ interval_minutes: 30 }),
    });
  } finally {
    setTimeout(() => { button.disabled = false; refreshAuto(); }, 1200);
  }
});

document.getElementById("startRun")?.addEventListener("click", async (event) => {
  const button = event.currentTarget;
  button.disabled = true;
  button.textContent = "Starting…";
  try {
    await fetch("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
  } finally {
    setTimeout(() => {
      button.disabled = false;
      button.textContent = "Train a new model";
      refresh();
    }, 2500);
  }
});

poll(refresh, POLL_MS);
poll(refreshAuto, POLL_MS);
