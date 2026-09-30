// Drawing a run so that every number can be argued with.
//
// Built with createElement throughout. Symbol names and error strings arrive
// from yfinance and from the news providers, and a page that puts those through
// innerHTML is a page that will one day render whatever a data provider decided
// to put in a field.

const POLL_MS = 15000;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function percent(value, digits = 1) {
  if (value === null || value === undefined) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

function signed(value, digits = 1) {
  if (value === null || value === undefined) return "—";
  const shown = (value * 100).toFixed(digits);
  return value > 0 ? `+${shown}%` : `${shown}%`;
}

// A plain signed number. A t-statistic is not a percentage, and running one
// through `signed` printed the gate's own hurdle as "-318.50%".
function value(number, digits = 2) {
  if (number === null || number === undefined) return "—";
  const shown = Number(number).toFixed(digits);
  return Number(number) > 0 ? `+${shown}` : shown;
}

// --- the pieces of a run ---------------------------------------------------

function statusPill(status) {
  const label = {
    building: "Building dataset",
    queued: "Waiting for a GPU",
    training: "Training",
    done: "Done",
    failed: "Failed",
  }[status] || status;

  const tone = status === "done" ? "is-ok"
    : status === "failed" ? "is-bad"
      : "is-working";
  return el("span", `pill ${tone}`, label);
}

function progressLine(run) {
  // A run spends minutes between the numbers being requested and the numbers
  // existing. Saying what it is doing is the difference between waiting and
  // wondering whether it has died.
  if (!run.progress || run.status === "done" || run.status === "failed") {
    return null;
  }
  const silent = run.silent_for || 0;
  const line = el("p", silent > 300 ? "note is-bad" : "note",
    silent > 300
      ? `${run.progress} — nothing for ${Math.round(silent / 60)} minutes, `
        + "which usually means the process running it is gone"
      : run.progress + "…");
  return line;
}

function figure(label, value, hint) {
  const box = el("div", "figure");
  box.appendChild(el("div", "figure-label", label));
  box.appendChild(el("div", "figure-value", value));
  if (hint) box.appendChild(el("div", "figure-hint", hint));
  return box;
}

function datasetPanel(run) {
  const data = run.dataset || {};
  const train = data.train || {};
  const test = data.test || {};

  const panel = el("section", "panel");
  panel.appendChild(el("h3", null, "What it learned from"));

  const grid = el("div", "figures");
  grid.appendChild(figure("Training rows", (train.rows || 0).toLocaleString(),
    `${train.from || "?"} → ${train.to || "?"}`));
  grid.appendChild(figure("Held back here", (test.rows || 0).toLocaleString(),
    `${test.from || "?"} → ${test.to || "?"}`));
  grid.appendChild(figure("Features", (data.feature_names || []).length,
    "per day, per symbol"));
  grid.appendChild(figure("Baseline", percent(test.up_share),
    "always guessing the commoner way"));
  panel.appendChild(grid);

  // The split is the single thing most likely to make the rest meaningless,
  // so it is stated rather than implied.
  if (train.to && test.from) {
    const ok = train.to < test.from;
    panel.appendChild(el("p", ok ? "note" : "note is-bad",
      ok
        ? `Split at ${data.cut_date || test.from}: everything the model trained on happened before everything it was graded on.`
        : "The training and test windows overlap. Any score below is worthless."));
  }
  return panel;
}

function evaluationPanel(run) {
  const evaluation = run.evaluation || {};
  if (!evaluation.rows) return null;

  const panel = el("section", "panel");
  panel.appendChild(el("h3", null, "What it scored on days it never saw"));

  const relative = (run.spec || {}).target === "relative";

  const grid = el("div", "figures");
  // Rows and days are different numbers. Labelling 4,514 symbol-days as
  // "4,514 days" overstated the sample by an order of magnitude in the one
  // place a reader is deciding how much to believe.
  grid.appendChild(figure("Accuracy", percent(evaluation.accuracy),
    evaluation.days
      ? `${evaluation.days.toLocaleString()} days, `
        + `${(evaluation.rows || 0).toLocaleString()} rows`
      : `${(evaluation.rows || 0).toLocaleString()} rows`));
  grid.appendChild(figure("Baseline", percent(evaluation.baseline_accuracy),
    evaluation.baseline_source || "the commoner direction"));
  grid.appendChild(figure("Edge", signed(evaluation.edge),
    "accuracy minus baseline"));
  grid.appendChild(figure("Says “up”", percent(evaluation.up_rate),
    "a stuck model sits near 0 or 100"));
  grid.appendChild(figure("Strategy", signed(evaluation.strategy_annualised),
    `annualised, ${percent(evaluation.cost_per_trade, 2)} per unit traded`));
  // A relative model is graded on market-relative returns, so an equal-weight
  // long book earns zero by construction. Calling that "just holding" invites
  // reading a structural zero as a result.
  grid.appendChild(relative
    ? figure("Turnover", (evaluation.turnover_daily || 0).toFixed(2),
      "of the book, per day")
    : figure("Just holding", signed(evaluation.hold_annualised),
      "same days, no trading"));
  panel.appendChild(grid);

  // The uncertainty that used to be missing entirely. The gate refuses to
  // speak about accuracy without a standard error; these are the same
  // discipline applied to the number somebody would actually act on.
  const risk = el("div", "figures");
  risk.appendChild(figure("Sharpe", (evaluation.strategy_sharpe ?? 0).toFixed(2),
    "close to close — not tradeable"));
  risk.appendChild(figure("t", (evaluation.strategy_tstat ?? 0).toFixed(2),
    "under 2 is not distinguishable from luck"));
  risk.appendChild(figure("Worst drawdown", percent(evaluation.strategy_max_drawdown),
    "peak to trough"));
  risk.appendChild(figure("Cost drag", percent(evaluation.cost_drag_annualised),
    `${(evaluation.position_changes || 0).toLocaleString()} position changes`));
  panel.appendChild(risk);

  // And the same money over the window an order could actually reach. Shown
  // as its own row rather than a column, because it is the one the gate reads
  // and the one a person would be acting on.
  if (evaluation.executable_sharpe !== undefined
      && evaluation.executable_sharpe !== null) {
    panel.appendChild(el("h4", null, "Held from the first open after the signal"));
    const real = el("div", "figures");
    real.appendChild(figure("Sharpe", (evaluation.executable_sharpe ?? 0).toFixed(2),
      "what an order could reach"));
    real.appendChild(figure("t", (evaluation.executable_tstat ?? 0).toFixed(2),
      "the gate reads this one"));
    real.appendChild(figure("Annualised", signed(evaluation.executable_annualised),
      "after costs"));
    real.appendChild(figure("Execution gap",
      (evaluation.execution_gap ?? 0).toFixed(2),
      "Sharpe lost to the overnight move"));
    panel.appendChild(real);

    panel.appendChild(el("p", "note",
      "The features are computed from a close, so nothing can be positioned at "
      + "that close on the strength of it — the earliest an order can go in is "
      + "the next open. The row above grades close to close, which is what the "
      + "label describes and what nobody can hold. This row grades open to "
      + "close. When they disagree, the gap is the overnight move, and it "
      + "belongs to whoever was already holding."));
  }

  if (evaluation.verdict !== undefined || run.verdict) {
    panel.appendChild(el("p", "verdict", run.verdict));
  }

  const buckets = (evaluation.by_confidence || []).filter((b) => b.rows > 0);
  if (buckets.length) {
    panel.appendChild(el("h4", null, "By how sure it was"));
    const table = el("table", "table");
    const head = el("tr");
    ["Confidence", "Rows", "Accuracy", "Range", "Close → close", "Open → close"].forEach((h) =>
      head.appendChild(el("th", null, h)));
    table.appendChild(head);

    buckets.forEach((bucket) => {
      const row = el("tr");
      row.appendChild(el("td", null, `${bucket.from}–${bucket.to}`));
      row.appendChild(el("td", null, bucket.rows.toLocaleString()));
      // Withheld below the minimum sample. "100% accurate on its five most
      // confident days" was the most misleading line this page ever printed.
      row.appendChild(el("td", null,
        bucket.enough ? percent(bucket.accuracy) : "too few"));
      const [low, high] = bucket.accuracy_interval || [null, null];
      row.appendChild(el("td", null,
        low === null ? "—" : `${percent(low)} – ${percent(high)}`));
      row.appendChild(el("td", null, signed(bucket.mean_net_return, 3)));
      row.appendChild(el("td", bucket.mean_executable_return < 0 ? "is-down" : null,
        bucket.mean_executable_return === null || bucket.mean_executable_return === undefined
          ? "—" : signed(bucket.mean_executable_return, 3)));
      table.appendChild(row);
    });
    panel.appendChild(table);
    panel.appendChild(el("p", "note",
      "A model worth anything is better on the days it commits. Flat across "
      + "these rows means the confidence number carries no information. "
      + "Accuracy is withheld on buckets too small to mean anything, and the "
      + "range beside it is how wide the honest interval actually is. The last "
      + "column is the only one an order can reach: on this panel the most "
      + "confident rows are 71.8% right and still lose money held from the "
      + "next open, because the move happens overnight."));
  }

  return panel;
}

function signalsPanel(run) {
  if (!run.signals || !run.signals.length) return null;

  const panel = el("section", "panel");
  panel.appendChild(el("h3", null, "Where it leans today"));

  const table = el("table", "table");
  const head = el("tr");
  ["Symbol", "As of", "Close", "Leaning", "P(up)", "How far from a coin flip"]
    .forEach((h) => head.appendChild(el("th", null, h)));
  table.appendChild(head);

  run.signals.forEach((signal) => {
    const row = el("tr");
    row.appendChild(el("td", "mono", signal.symbol));
    row.appendChild(el("td", null, signal.as_of));
    row.appendChild(el("td", "mono", signal.close));
    row.appendChild(el("td", signal.leaning === "up" ? "up" : "down",
      signal.leaning));
    row.appendChild(el("td", "mono", signal.probability_up.toFixed(3)));

    const bar = el("td");
    const track = el("div", "bar");
    const fill = el("div", "bar-fill");
    fill.style.width = `${Math.round(signal.confidence * 100)}%`;
    track.appendChild(fill);
    bar.appendChild(track);
    bar.appendChild(el("span", "bar-label", percent(signal.confidence, 0)));
    row.appendChild(bar);

    table.appendChild(row);
  });

  panel.appendChild(table);
  panel.appendChild(el("p", "note",
    "P(up) near 0.5 means the model has no opinion, which is the honest answer "
    + "most days. Nothing here is an instruction to buy or sell."));
  return panel;
}

// Why the gate said what it said, in a few words. The full reason is a
// paragraph and belongs underneath; the card needs the conclusion.
const WHY = {
  makes_money: "accurate, but it loses money",
  beats_chance: "its edge is inside what chance produces",
  beats_baseline: "no better than always guessing the common direction",
  beats_noise_floor: "its edge is smaller than the spread between seeds",
  consistent_across_time: "its edge shows up in one window, not most",
  model_varies: "it answers the same way nearly every day",
  enough_days: "too few days graded to say anything",
  enough_effective_rows: "too little independent evidence",
};

function verdictLine(run) {
  const evaluation = run.evaluation || {};
  if (!evaluation.rows) return null;

  const trust = run.trust || {};
  const failed = (trust.checks || []).find((check) => !check.passed);
  const why = failed ? (WHY[failed.name] || failed.name.replace(/_/g, " ")) : "";

  const line = el("p", "run-verdict");
  line.appendChild(el("span", trust.trusted ? "up" : "down",
    trust.trusted ? "Gate open" : "Gate shut"));
  line.appendChild(el("span", null, why ? ` — ${why}.` : "."));
  line.appendChild(el("span", "run-numbers",
    `${signed(evaluation.edge)} edge · open→close t `
    + `${value(evaluation.executable_tstat)} · ${evaluation.days || 0} days graded`));
  return line;
}

function runCard(run) {
  const card = el("article", "run");

  const head = el("div", "run-head");
  const title = el("div");
  title.appendChild(el("h2", null, run.run_id));
  title.appendChild(el("div", "run-sub",
    `${run.watchlist.length} symbols · ${run.horizon}-day horizon`
    + (run.backend ? ` · ${run.backend}` : "")));
  head.appendChild(title);
  head.appendChild(statusPill(run.status));
  card.appendChild(head);

  if (run.error) card.appendChild(el("p", "error", run.error));

  const progress = progressLine(run);
  if (progress) card.appendChild(progress);

  const verdict = verdictLine(run);
  if (verdict) card.appendChild(verdict);
  if (run.verdict) card.appendChild(el("p", "run-reading", run.verdict));

  const panels = [datasetPanel(run), evaluationPanel(run),
                  signalsPanel(run)].filter(Boolean);
  if (panels.length) {
    const fold = el("details", "fold");
    fold.appendChild(el("summary", null, "The working"));
    panels.forEach((panel) => fold.appendChild(panel));
    card.appendChild(fold);
  }

  return card;
}

// --- keeping it current ----------------------------------------------------

async function refresh() {
  try {
    // Ask the server to pick up anything that finished while we were away.
    // Safe to repeat: it only acts on runs that are not already resolved.
    await fetch("/api/collect", { method: "POST" }).catch(() => {});

    const response = await fetch("/api/runs");
    if (!response.ok) throw new Error(`server returned ${response.status}`);
    const runs = await response.json();

    const host = document.getElementById("runs");
    host.replaceChildren();

    if (!runs.length) {
      host.appendChild(el("p", "empty",
        "No runs yet. Train a model to see what it does."));
      return;
    }
    runs.forEach((run) => host.appendChild(runCard(run)));
  } catch (error) {
    const host = document.getElementById("runs");
    host.replaceChildren(el("p", "error", `Could not load runs: ${error.message}`));
  }
}

function describeAuto(state) {
  if (!state.running) {
    if (state.stopped_because) return `Auto-training off — ${state.stopped_because}`;
    if (state.stopped_at) return `Auto-training off since ${state.stopped_at.replace("T", " ")}`;
    return "Auto-training off";
  }

  const last = state.last || {};
  const did = last.error ? `last cycle failed: ${last.error}`
    : last.trained ? `trained ${last.trained}`
    : last.skipped ? "idle — no new session to train on"
    : "starting";
  const checks = `${state.cycles || 0} check${state.cycles === 1 ? "" : "s"}`;
  return `Auto-training on, every ${state.interval_minutes} min — ${checks}, ${did}`;
}

async function refreshAuto() {
  try {
    const response = await fetch("/api/auto");
    if (!response.ok) return;
    const state = await response.json();

    const button = document.getElementById("autoTrain");
    button.textContent = state.running ? "Stop auto-training" : "Auto-train";
    button.dataset.running = state.running ? "yes" : "no";
    button.classList.toggle("is-running", Boolean(state.running));

    const line = document.getElementById("autoState");
    line.textContent = describeAuto(state);
    line.className = state.running ? "auto-state is-running" : "auto-state";
  } catch {
    /* the runs panel already reports a dead server */
  }
}

document.getElementById("autoTrain").addEventListener("click", async (event) => {
  const button = event.currentTarget;
  const stopping = button.dataset.running === "yes";
  button.disabled = true;

  try {
    const response = await fetch(stopping ? "/api/auto/stop" : "/api/auto/start",
      { method: "POST" });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.error || `server returned ${response.status}`);
  } catch (error) {
    document.getElementById("autoState").textContent = error.message;
  } finally {
    button.disabled = false;
    await refreshAuto();
  }
});

async function refreshStatus() {
  try {
    const response = await fetch("/api/status");
    if (!response.ok) return;
    const status = await response.json();
    const universe = document.getElementById("universeCount");
    if (universe) universe.textContent = `${status.universe} symbols`;
  } catch {
    /* the runs panel already reports a dead server */
  }
}

document.getElementById("startRun").addEventListener("click", async (event) => {
  const button = event.currentTarget;
  button.disabled = true;
  button.textContent = "Building…";

  try {
    const response = await fetch("/api/runs", { method: "POST" });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.error || `server returned ${response.status}`);
    await refresh();
  } catch (error) {
    const host = document.getElementById("runs");
    host.prepend(el("p", "error", error.message));
  } finally {
    button.disabled = false;
    button.textContent = "Train a new model";
  }
});

refresh();
refreshStatus();
refreshAuto();
setInterval(refresh, POLL_MS);
setInterval(refreshStatus, POLL_MS);
setInterval(refreshAuto, POLL_MS);
