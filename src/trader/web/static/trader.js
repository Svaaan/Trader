// Drawing a run so that every number can be argued with.
//
// Built with createElement throughout. Symbol names and error strings arrive
// from yfinance and from the coordinator, and a page that puts those through
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

function verificationPanel(run) {
  const verification = run.verification || {};
  const measured = verification.measured || {};
  if (!verification.verdict) return null;

  const panel = el("section", "panel");
  panel.appendChild(el("h3", null, "What HelloWorldAi checked"));

  const grid = el("div", "figures");
  grid.appendChild(figure("Verdict", verification.verdict,
    verification.strength || ""));
  grid.appendChild(figure("Holdout accuracy", percent(measured.holdout_accuracy),
    "on a random slice"));
  grid.appendChild(figure("Untrained model", percent(measured.untrained_accuracy),
    "the floor it had to beat"));
  panel.appendChild(grid);

  panel.appendChild(el("p", "note",
    "The coordinator rebuilt the model from the returned weights and scored it "
    + "itself, so a node cannot report a result it did not get. Its slice is "
    + "random, though, which for a price series is easier than predicting "
    + "forwards — read it as proof that training happened, not as skill."));
  return panel;
}

function evaluationPanel(run) {
  const evaluation = run.evaluation || {};
  if (!evaluation.rows) return null;

  const panel = el("section", "panel");
  panel.appendChild(el("h3", null, "What it scored on days it never saw"));

  const grid = el("div", "figures");
  grid.appendChild(figure("Accuracy", percent(evaluation.accuracy),
    `${evaluation.rows.toLocaleString()} days`));
  grid.appendChild(figure("Baseline", percent(evaluation.baseline_accuracy),
    "always the commoner direction"));
  grid.appendChild(figure("Edge", signed(evaluation.edge),
    "accuracy minus baseline"));
  grid.appendChild(figure("Says “up”", percent(evaluation.up_rate),
    "a stuck model sits near 0 or 100"));
  grid.appendChild(figure("Strategy", signed(evaluation.strategy_annualised),
    `annualised, after ${percent(evaluation.cost_per_trade, 2)} a trade`));
  grid.appendChild(figure("Just holding", signed(evaluation.hold_annualised),
    "same days, no trading"));
  panel.appendChild(grid);

  if (evaluation.verdict !== undefined || run.verdict) {
    panel.appendChild(el("p", "verdict", run.verdict));
  }

  const buckets = (evaluation.by_confidence || []).filter((b) => b.days > 0);
  if (buckets.length) {
    panel.appendChild(el("h4", null, "By how sure it was"));
    const table = el("table", "table");
    const head = el("tr");
    ["Confidence", "Days", "Accuracy", "Mean return"].forEach((h) =>
      head.appendChild(el("th", null, h)));
    table.appendChild(head);

    buckets.forEach((bucket) => {
      const row = el("tr");
      row.appendChild(el("td", null, `${bucket.from}–${bucket.to}`));
      row.appendChild(el("td", null, bucket.days.toLocaleString()));
      row.appendChild(el("td", null, percent(bucket.accuracy)));
      row.appendChild(el("td", null, signed(bucket.mean_return, 3)));
      table.appendChild(row);
    });
    panel.appendChild(table);
    panel.appendChild(el("p", "note",
      "A model worth anything is better on the days it commits. Flat across "
      + "these rows means the confidence number carries no information."));
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

function runCard(run) {
  const card = el("article", "run");

  const head = el("div", "run-head");
  const title = el("div");
  title.appendChild(el("h2", null, run.run_id));
  title.appendChild(el("div", "run-sub",
    `${run.watchlist.length} symbols · ${run.horizon}-day horizon`
    + (run.task_id ? ` · job ${run.task_id.slice(0, 16)}…` : "")));
  head.appendChild(title);
  head.appendChild(statusPill(run.status));
  card.appendChild(head);

  if (run.error) {
    card.appendChild(el("p", "error", run.error));
  }

  [datasetPanel(run), verificationPanel(run), evaluationPanel(run),
   signalsPanel(run)]
    .filter(Boolean)
    .forEach((panel) => card.appendChild(panel));

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

async function refreshStatus() {
  try {
    const response = await fetch("/api/status");
    if (!response.ok) return;
    const status = await response.json();
    const pill = document.getElementById("coordinatorPill");
    pill.textContent = status.coordinator_detail;
    pill.className = `pill ${status.coordinator_ok ? "is-ok" : "is-bad"}`;
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
setInterval(refresh, POLL_MS);
setInterval(refreshStatus, POLL_MS);
