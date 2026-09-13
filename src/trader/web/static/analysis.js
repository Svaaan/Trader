// The trading view: a call per symbol, and the argument for it.
//
// The order on the page is the order of the reasoning, and it is deliberate.
// First whether the model has earned an opinion at all; then what it had to
// beat; then whether it held up across time; then what it pays attention to;
// then, underneath all of it, what it says about each symbol. Putting the calls
// first would make the gate look like a footnote on a recommendation rather
// than the thing that decides whether there is one.
//
// createElement throughout: symbol names and headlines come from providers.

const POLL_MS = 30000;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

const pct = (v, d = 1) => (v === null || v === undefined ? "—" : `${(v * 100).toFixed(d)}%`);
const num = (v, d = 2) => (v === null || v === undefined ? "—" : Number(v).toFixed(d));

const VERDICT = {
  buy: { label: "Buy now", tone: "is-buy" },
  no_buy: { label: "No buy", tone: "is-nobuy" },
  unsure: { label: "Still collecting data", tone: "is-unsure" },
};

function figure(label, value, hint) {
  const box = el("div", "figure");
  box.appendChild(el("div", "figure-label", label));
  box.appendChild(el("div", "figure-value", value));
  if (hint) box.appendChild(el("div", "figure-hint", hint));
  return box;
}

// --- has it earned an opinion? --------------------------------------------

function trustPanel(run) {
  const trust = run.trust || {};
  const evaluation = run.evaluation || {};

  const panel = el("section", `a-panel trust ${trust.trusted ? "is-ok" : "is-bad"}`);

  const head = el("div", "trust-head");
  head.appendChild(el("h2", null,
    trust.trusted ? "This model has shown an edge" : "This model has not shown an edge"));
  head.appendChild(el("span", `pill ${trust.trusted ? "is-ok" : "is-bad"}`,
    trust.trusted ? "calls enabled" : "calls suppressed"));
  panel.appendChild(head);

  panel.appendChild(el("p", "trust-reason", trust.reason || ""));

  const grid = el("div", "figures");
  // Runs from before the distinction existed have rows and no day count, and
  // showing "0 days" would be worse than showing the number they do have.
  grid.appendChild(figure("Accuracy", pct(evaluation.accuracy),
    evaluation.days
      ? `${evaluation.days.toLocaleString()} days it never saw`
      : `${(evaluation.rows || 0).toLocaleString()} rows it never saw`));
  grid.appendChild(figure("Baseline", pct(evaluation.baseline_accuracy),
    // Only claim the training majority when the run actually recorded that it
    // used one; older runs read the baseline off the test period.
    evaluation.baseline_source || "the commoner direction"));
  grid.appendChild(figure("Edge", pct(trust.edge, 2), "accuracy minus baseline"));
  grid.appendChild(figure("Needed", pct(trust.needed, 2),
    "two standard errors, from the effective sample"));
  panel.appendChild(grid);

  // The row count and the sample size are different numbers, and the gap
  // between them is the whole reason the bar sits where it does -- when there
  // is a gap. A relative target largely removes it, and claiming a discount
  // that was not applied would be its own small dishonesty.
  if (evaluation.effective_rows) {
    const rows = (evaluation.rows || 0).toLocaleString();
    const effective = evaluation.effective_rows.toLocaleString();
    const discounted = evaluation.effective_rows < (evaluation.rows || 0);

    panel.appendChild(el("p", "note",
      discounted
        ? `${rows} rows across ${evaluation.symbols || "?"} symbols, but they `
          + `move together (mean correlation ${num(evaluation.row_correlation, 3)}), `
          + `so they are worth about ${effective} independent observations. The `
          + `bar is computed from that smaller number, because ten symbols on `
          + `one day are not ten verdicts.`
        : `${rows} rows across ${evaluation.symbols || "?"} symbols, and this `
          + `time they do not move together (mean correlation `
          + `${num(evaluation.row_correlation, 3)}), so all of them count. A `
          + `relative target tends to do that: once the market move is taken `
          + `out of the label, being right about one name says little about `
          + `being right about the next.`));
  }

  // The two holding windows, side by side, whenever they disagree. This is
  // the difference between a backtest and a trade.
  if (evaluation.executable_sharpe !== undefined
      && evaluation.executable_sharpe !== null) {
    const money = el("table", "compare");
    const header = el("tr");
    ["Held", "Sharpe", "t", "Annualised"].forEach((label) => {
      header.appendChild(el("th", null, label));
    });
    money.appendChild(header);

    [["close(t) → close(t+1) — what the label means",
      evaluation.strategy_sharpe, evaluation.strategy_tstat,
      evaluation.strategy_annualised, false],
     ["open(t+1) → close(t+1) — what an order reaches",
      evaluation.executable_sharpe, evaluation.executable_tstat,
      evaluation.executable_annualised, true],
    ].forEach(([label, sharpe, tstat, ann, isSubject]) => {
      const row = el("tr", isSubject ? "is-subject" : null);
      row.appendChild(el("td", null, label));
      row.appendChild(el("td", null, num(sharpe)));
      row.appendChild(el("td", null, num(tstat)));
      row.appendChild(el("td", null, pct(ann)));
      money.appendChild(row);
    });
    panel.appendChild(money);

    if (Math.abs(evaluation.execution_gap || 0) > 0.2) {
      panel.appendChild(el("p", "warn",
        `${num(evaluation.execution_gap)} of Sharpe separates the two. The `
        + "features are computed from a close, so nothing can be positioned at "
        + "that close on the strength of it — the gap is the overnight move, "
        + "and it belongs to whoever was already holding. The gate reads the "
        + "second row."));
    }
  }

  // Every hurdle, passed or not. A gate that says no without saying which
  // question it failed teaches nobody anything.
  const checks = trust.checks || [];
  if (checks.length) {
    const list = el("ul", "checks");
    checks.forEach((check) => {
      const row = el("li", check.passed ? "is-ok" : "is-bad");
      row.appendChild(el("span", "check-mark", check.passed ? "✓" : "✗"));
      row.appendChild(el("span", "check-name", check.name.replace(/_/g, " ")));
      row.appendChild(el("span", "check-detail", check.detail));
      list.appendChild(row);
    });
    panel.appendChild(list);
  }

  return panel;
}

// --- what it had to beat ---------------------------------------------------

function controlsPanel(run) {
  const controls = run.controls || {};
  const evaluation = run.evaluation || {};
  if (!controls.majority) return null;

  const panel = el("section", "a-panel");
  panel.appendChild(el("h2", null, "What it had to beat"));

  const rows = [
    ["Always the training majority", controls.majority],
    ["Logistic regression, same rows", controls.logistic],
    ["Local MLP, same shape", controls.local_mlp],
    ["HelloWorldAi model", evaluation],
  ].filter(([, value]) => value && value.accuracy !== undefined);

  const table = el("table", "compare");
  const header = el("tr");
  ["Model", "Accuracy", "Edge", "Up-rate", "Sharpe", "t"].forEach((label) => {
    header.appendChild(el("th", null, label));
  });
  table.appendChild(header);

  rows.forEach(([label, result], index) => {
    const row = el("tr", index === rows.length - 1 ? "is-subject" : null);
    row.appendChild(el("td", null, label));
    row.appendChild(el("td", null, pct(result.accuracy, 2)));
    row.appendChild(el("td", null, pct(result.edge, 2)));
    row.appendChild(el("td", null, pct(result.up_rate, 0)));
    row.appendChild(el("td", null, num(result.strategy_sharpe)));
    row.appendChild(el("td", null, num(result.strategy_tstat)));
    table.appendChild(row);
  });
  panel.appendChild(table);

  const floor = controls.noise_floor;
  if (floor && floor.seeds > 1) {
    panel.appendChild(figureRow(
      "Noise floor",
      pct(floor.spread, 2),
      `${floor.seeds} identical configurations, different seeds only`));
    panel.appendChild(el("p", "note",
      "That spread is the smallest difference between two models that means "
      + "anything. An edge smaller than it is a fact about which seed came up, "
      + "and the gate above refuses it on exactly that basis."));
  }

  const hyper = controls.hyperparameters;
  let how = "All three ran here, on the same rows. ";
  if (hyper) {
    how = `All three ran here, on the same rows, with the submission's own `
      + `shape (${hyper.hidden} wide, ${hyper.depth} deep). `;
    // A control that saw a quarter of the steps is worth having and is not the
    // same claim as one that saw all of them, so the gap is stated.
    if (controls.local_step_share !== undefined && controls.local_step_share < 1) {
      how += `The local MLP was trained for ${(controls.local_steps || 0).toLocaleString()} `
        + `of the submission's ${hyper.steps.toLocaleString()} steps `
        + `(${pct(controls.local_step_share, 0)}) to keep it to seconds, so read `
        + `it as a floor rather than a match. `;
    } else if (hyper.steps) {
      how += `The local MLP saw the same ${hyper.steps.toLocaleString()} steps. `;
    }
  }

  panel.appendChild(el("p", "note",
    how
    + "If the trained model does not beat the logistic regression, the round "
    + "trip bought a linear model slowly; if it does not beat the majority, it "
    + "learnt nothing."));

  return panel;
}

function figureRow(label, value, hint) {
  const grid = el("div", "figures");
  grid.appendChild(figure(label, value, hint));
  return grid;
}

// --- the two backends ------------------------------------------------------

function backendPanel(run) {
  const local = run.local_evaluation || {};
  const comparison = run.comparison || {};
  if (!local.accuracy && !comparison.gap) return null;

  const panel = el("section", "a-panel");
  panel.appendChild(el("h2", null, "Trained here, and trained there"));

  const rows = [
    ["Trained here (numpy, no GPU)", local, run.primary === "local"],
    ["HelloWorldAi", run.evaluation || {}, run.primary === "helloworld"],
  ].filter(([, value]) => value && value.accuracy !== undefined);

  const table = el("table", "compare");
  const header = el("tr");
  ["Where", "Accuracy", "Edge", "Up-rate", "Sharpe", "t"].forEach((label) => {
    header.appendChild(el("th", null, label));
  });
  table.appendChild(header);

  rows.forEach(([label, result, isPrimary]) => {
    const row = el("tr", isPrimary ? "is-subject" : null);
    row.appendChild(el("td", null, label));
    row.appendChild(el("td", null, pct(result.accuracy, 2)));
    row.appendChild(el("td", null, pct(result.edge, 2)));
    row.appendChild(el("td", null, pct(result.up_rate, 0)));
    row.appendChild(el("td", null, num(result.strategy_sharpe)));
    row.appendChild(el("td", null, num(result.strategy_tstat)));
    table.appendChild(row);
  });
  panel.appendChild(table);

  if (comparison.gap !== undefined && comparison.gap !== null) {
    panel.appendChild(figureRow(
      "Gap",
      pct(comparison.gap, 2),
      comparison.multiples_of_noise !== undefined
        ? `${num(comparison.multiples_of_noise)}× the seed spread`
        : "no noise floor measured"));
  }

  // The reading is the point. A gap in points means nothing without knowing
  // how far apart two identical models land by chance.
  panel.appendChild(el("p", "note", comparison.reading
    || "Only one backend has reported so far. The comparison appears when the "
       + "other does."));

  panel.appendChild(el("p", "note",
    "Both were given the same rows and the same hyperparameters, and both are "
    + "loaded by the same numpy forward pass and scored by the same evaluator. "
    + "So a difference between them is a fact about the round trip — placement, "
    + "the trainer, the holdout it carves out, the weights that came back — "
    + "rather than about the data. The model is 7,233 parameters; training it "
    + "here takes about half a minute and needs no GPU."));

  return panel;
}

// --- did it hold up across time? -------------------------------------------

function walkForwardPanel(run) {
  const wf = run.walk_forward || {};
  const folds = wf.folds || [];
  if (!folds.length) return null;

  const panel = el("section", "a-panel");
  panel.appendChild(el("h2", null, "Across consecutive windows"));

  const table = el("table", "compare");
  const header = el("tr");
  ["Window", "Rows", "Accuracy", "Baseline", "Edge", "Sharpe"].forEach((label) => {
    header.appendChild(el("th", null, label));
  });
  table.appendChild(header);

  folds.forEach((fold) => {
    const row = el("tr", fold.edge > 0 ? "is-ok" : "is-bad");
    row.appendChild(el("td", null, `${fold.test_from} → ${fold.test_to}`));
    row.appendChild(el("td", null, (fold.rows || 0).toLocaleString()));
    row.appendChild(el("td", null, pct(fold.accuracy, 2)));
    row.appendChild(el("td", null, pct(fold.baseline, 2)));
    row.appendChild(el("td", null, pct(fold.edge, 2)));
    row.appendChild(el("td", null, num(fold.sharpe)));
    table.appendChild(row);
  });
  panel.appendChild(table);

  panel.appendChild(el("p", "note",
    `Positive in ${wf.folds_positive} of ${wf.folds_run} windows, mean edge `
    + `${pct(wf.mean_edge, 2)} with a spread of ${pct(wf.sd_edge, 2)}. `
    + "One test period is one draw. An edge that shows up in a single window "
    + "and not the others is what a coin looks like, and the gate treats it "
    + `that way. Graded with the ${wf.model} control, because six GPU round `
    + "trips would take a day and this takes seconds."));

  return panel;
}

// --- what went into it -----------------------------------------------------

function datasetPanel(run) {
  const dataset = run.dataset || {};
  const blocks = dataset.blocks || {};
  const excluded = dataset.excluded || {};

  const panel = el("section", "a-panel");
  panel.appendChild(el("h2", null, "What went in"));

  const grid = el("div", "figures");
  grid.appendChild(figure("Symbols", (dataset.symbols || []).length,
    `${Object.keys(excluded).length} excluded`));
  grid.appendChild(figure("Features", (dataset.feature_names || []).length,
    "own, cross-sectional, macro, events"));
  grid.appendChild(figure("Train rows", (dataset.train?.rows || 0).toLocaleString(),
    `to ${dataset.train?.to || "?"}`));
  grid.appendChild(figure("Test rows", (dataset.test?.rows || 0).toLocaleString(),
    `from ${dataset.test?.from || "?"}`));
  panel.appendChild(grid);

  const chips = el("div", "chips");
  Object.entries(blocks).forEach(([name, block]) => {
    if (!block) return;
    const used = block.used;
    const chip = el("span", `chip ${used ? "is-ok" : "is-off"}`,
      `${name}: ${used ? `${(block.columns || []).length} columns` : "off"}`);
    chips.appendChild(chip);
  });
  panel.appendChild(chips);

  // A watchlist of ten that trains as nine used to be invisible.
  const names = Object.keys(excluded);
  if (names.length) {
    const list = el("ul", "excluded");
    names.slice(0, 12).forEach((symbol) => {
      list.appendChild(el("li", null, `${symbol} — ${excluded[symbol]}`));
    });
    if (names.length > 12) {
      list.appendChild(el("li", null, `…and ${names.length - 12} more`));
    }
    panel.appendChild(el("h3", null, "Left out of the panel"));
    panel.appendChild(list);
  }

  if (dataset.panel_drift) {
    panel.appendChild(el("p", "warn",
      `The scoring panel differed from the trained one: added `
      + `${(dataset.panel_drift.added || []).join(", ") || "none"}, lost `
      + `${(dataset.panel_drift.lost || []).join(", ") || "none"}. Only the `
      + "trained symbols were graded."));
  }

  return panel;
}

// --- what it pays attention to --------------------------------------------

function learntPanel(run) {
  const learnt = run.learnt || [];
  if (!learnt.length) return null;

  const panel = el("section", "a-panel");
  panel.appendChild(el("h2", null, "What it learnt to look at"));

  const strongest = Math.max(...learnt.map((f) => f.influence), 1e-9);

  const list = el("div", "influence");
  learnt.slice(0, 20).forEach((feature) => {
    const row = el("div", "influence-row");
    row.appendChild(el("div", "influence-name", feature.feature.replace(/_/g, " ")));

    const track = el("div", "bar");
    const fill = el("div", "bar-fill");
    fill.style.width = `${Math.max(2, Math.round((feature.influence / strongest) * 100))}%`;
    // Which way it usually pushes, which is the readable half of the number.
    if (feature.leans < 0) fill.classList.add("is-down");
    track.appendChild(fill);
    row.appendChild(track);

    row.appendChild(el("div", "influence-value",
      `${(feature.influence * 100).toFixed(2)} pts`));
    list.appendChild(row);
  });
  panel.appendChild(list);

  panel.appendChild(el("p", "note",
    "Measured by muting one input at a time — setting it to its training "
    + "average and seeing how far the answer moves. Bars near zero across the "
    + "board mean the model is barely reading its inputs, which is what a "
    + "network that has collapsed to one answer looks like from the inside."));

  return panel;
}

// --- the calls -------------------------------------------------------------

function signalCard(signal, gateOpen) {
  const verdict = VERDICT[signal.verdict] || VERDICT.unsure;

  const card = el("article", `call ${verdict.tone}`);

  const head = el("div", "call-head");
  const left = el("div");
  left.appendChild(el("span", "call-symbol", signal.symbol));
  // A date reads as a label; "11 days old" reads as the problem it is.
  const stale = signal.stale_days;
  left.appendChild(el("span",
    stale !== undefined && stale > 4 ? "call-asof is-stale" : "call-asof",
    stale !== undefined && stale > 4
      ? ` as of ${signal.as_of} — ${stale} days old`
      : ` as of ${signal.as_of}`));
  head.appendChild(left);
  head.appendChild(el("span", `call-verdict ${verdict.tone}`, verdict.label));
  card.appendChild(head);

  const numbers = el("div", "call-numbers");
  if (signal.close !== null && signal.close !== undefined) {
    numbers.appendChild(el("span", null, `Close ${signal.close}`));
  }
  numbers.appendChild(el("span", null, `P(up) ${signal.probability_up.toFixed(3)}`));
  card.appendChild(numbers);

  card.appendChild(el("p", "call-because", signal.because || ""));

  const reasons = signal.reasons || [];
  if (reasons.length) {
    const why = el("ul", "call-reasons");
    reasons.forEach((reason) => why.appendChild(el("li", null, reason.sentence)));
    card.appendChild(why);
  }

  // Background is fetched on demand, so a page of two hundred symbols does not
  // make two hundred calls nobody asked for.
  const more = el("button", "context-button", "Background");
  const slot = el("div", "context-slot");
  more.addEventListener("click", () => loadContext(signal.symbol, slot, more, gateOpen));
  card.appendChild(more);
  card.appendChild(slot);

  return card;
}

async function loadContext(symbol, slot, button, gateOpen) {
  button.disabled = true;
  button.textContent = "Reading…";
  slot.replaceChildren();

  try {
    const response = await fetch(`/api/context/${encodeURIComponent(symbol)}`);
    if (!response.ok) throw new Error(`server returned ${response.status}`);
    const body = await response.json();

    const box = el("div", `context ${gateOpen ? "" : "is-gated"}`);
    // The heading says what this is before the prose does, because prose reads
    // as conviction and this is not evidence.
    box.appendChild(el("div", "context-head",
      gateOpen ? "Context" : "Background — not evidence"));
    box.appendChild(el("p", "context-text", body.text));

    const sources = body.sources || [];
    if (sources.length) {
      const list = el("ol", "sources");
      sources.forEach((source) => {
        const item = el("li");
        if (source.url) {
          const link = el("a", null, source.title || source.url);
          link.href = source.url;
          link.target = "_blank";
          link.rel = "noopener noreferrer";
          item.appendChild(link);
        } else {
          item.appendChild(el("span", null, source.title || "untitled"));
        }
        // Capture time, not the provider's claim: it is the only timestamp
        // here that could not have been revised after the fact.
        item.appendChild(el("span", "source-meta",
          ` — ${source.provider || "unknown"}, first seen ${source.captured_utc}`));
        list.appendChild(item);
      });
      box.appendChild(list);
    }

    box.appendChild(el("p", "note", body.note || ""));
    slot.appendChild(box);

  } catch (error) {
    slot.appendChild(el("p", "error", `Could not load: ${error.message}`));
  } finally {
    button.disabled = false;
    button.textContent = "Background";
  }
}

function callsPanel(run) {
  const signals = run.signals || [];
  const gateOpen = Boolean((run.trust || {}).trusted);

  const panel = el("section", "a-panel");
  panel.appendChild(el("h2", null, "Today"));

  if (!signals.length) {
    panel.appendChild(el("p", "empty", "No signals in this run."));
    return panel;
  }

  const grid = el("div", "calls");
  signals.forEach((signal) => grid.appendChild(signalCard(signal, gateOpen)));
  panel.appendChild(grid);

  panel.appendChild(el("p", "note",
    "The reasons describe what moved this model's answer today. They explain "
    + "the decision; they are not evidence the decision is right, and they are "
    + "worth reading only when the gate above is open."));

  return panel;
}

// --- the news store --------------------------------------------------------

function newsPanel(body) {
  const state = body.news || {};
  if (!state.needed_days) return null;

  const panel = el("section", "a-panel");
  panel.appendChild(el("h2", null, "Point-in-time news store"));

  const grid = el("div", "figures");
  grid.appendChild(figure("Items", (state.items || 0).toLocaleString(),
    state.since ? `since ${String(state.since).slice(0, 10)}` : "nothing yet"));
  grid.appendChild(figure("History", `${state.history_days || 0} d`,
    `${state.needed_days} needed`));
  grid.appendChild(figure("Usable as features", state.ready ? "yes" : "not yet",
    state.ready ? "" : "accumulating"));
  panel.appendChild(grid);

  panel.appendChild(el("p", "note",
    "News cannot be backfilled honestly: an archive fetched today has been "
    + "re-ranked by what turned out to matter, and its publication timestamps "
    + "are often ingestion times. So the store records when this machine "
    + "provably saw each item and is written forwards only. Until it clears "
    + `${state.needed_days} days it is used for reading, never as a feature.`));

  if (!body.context_available) {
    panel.appendChild(el("p", "note",
      "No language model is configured, so the background panels show sources "
      + "without prose. Set ANTHROPIC_API_KEY to enable them."));
  }

  return panel;
}

// --- assembly --------------------------------------------------------------

async function refresh() {
  const host = document.getElementById("analysis");

  try {
    const response = await fetch("/api/analysis");
    if (!response.ok) throw new Error(`server returned ${response.status}`);
    const body = await response.json();

    host.replaceChildren();

    if (!body.run) {
      host.appendChild(el("p", "empty", body.why || "Nothing to show yet."));
      return;
    }

    const run = body.run;
    const target = run.spec?.target || "absolute";

    const where = run.primary === "local" ? "trained here"
      : run.primary === "helloworld" ? "trained on HelloWorldAi"
        : "";
    host.appendChild(el("p", "analysis-meta",
      `Model ${run.run_id}${where ? ", " + where : ""}, ${run.horizon}-day `
      + `horizon, ${target} target, graded on `
      + `${run.dataset?.test?.from || "?"} → ${run.dataset?.test?.to || "?"}.`));

    [
      trustPanel(run),
      backendPanel(run),
      controlsPanel(run),
      walkForwardPanel(run),
      datasetPanel(run),
      learntPanel(run),
      callsPanel(run),
      newsPanel(body),
    ].filter(Boolean).forEach((panel) => host.appendChild(panel));

  } catch (error) {
    host.replaceChildren(el("p", "error", `Could not load: ${error.message}`));
  }
}

refresh();
setInterval(refresh, POLL_MS);
