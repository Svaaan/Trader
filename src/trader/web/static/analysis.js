// The trading view: a call per symbol, and the argument for it.
//
// The order on the page is the order of the reasoning, and it is deliberate.
// First whether the model has earned an opinion at all; then what it pays
// attention to; then, underneath both, what it says about each symbol. Putting
// the calls first would make the gate look like a footnote on a recommendation
// rather than the thing that decides whether there is one.
//
// createElement throughout: symbol names come from a data provider.

const POLL_MS = 30000;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

const pct = (v, d = 1) => (v === null || v === undefined ? "—" : `${(v * 100).toFixed(d)}%`);

const VERDICT = {
  buy: { label: "Buy now", tone: "is-buy" },
  no_buy: { label: "No buy", tone: "is-nobuy" },
  unsure: { label: "Still collecting data", tone: "is-unsure" },
};

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
  const figure = (label, value, hint) => {
    const box = el("div", "figure");
    box.appendChild(el("div", "figure-label", label));
    box.appendChild(el("div", "figure-value", value));
    if (hint) box.appendChild(el("div", "figure-hint", hint));
    return box;
  };

  grid.appendChild(figure("Accuracy", pct(evaluation.accuracy),
    `${(evaluation.rows || 0).toLocaleString()} days it never saw`));
  grid.appendChild(figure("Baseline", pct(evaluation.baseline_accuracy),
    "always the commoner direction"));
  grid.appendChild(figure("Edge", pct(trust.edge, 2), "accuracy minus baseline"));
  grid.appendChild(figure("Needed", pct(trust.needed, 2),
    "two standard errors, from the sample size"));
  panel.appendChild(grid);

  // The bar is computed from how much evidence there is, not chosen, so it is
  // worth saying that out loud where somebody is deciding whether to believe it.
  panel.appendChild(el("p", "note",
    "The bar moves with the amount of evidence: over a few hundred days chance "
    + "alone can produce several points of apparent edge, over several thousand "
    + "it cannot. Nothing is called unless the gap clears it."));

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
  learnt.forEach((feature) => {
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

function signalCard(signal) {
  const verdict = VERDICT[signal.verdict] || VERDICT.unsure;

  const card = el("article", `call ${verdict.tone}`);

  const head = el("div", "call-head");
  const left = el("div");
  left.appendChild(el("span", "call-symbol", signal.symbol));
  left.appendChild(el("span", "call-asof", ` as of ${signal.as_of}`));
  head.appendChild(left);
  head.appendChild(el("span", `call-verdict ${verdict.tone}`, verdict.label));
  card.appendChild(head);

  const numbers = el("div", "call-numbers");
  numbers.appendChild(el("span", null, `Close ${signal.close}`));
  numbers.appendChild(el("span", null, `P(up) ${signal.probability_up.toFixed(3)}`));
  card.appendChild(numbers);

  card.appendChild(el("p", "call-because", signal.because || ""));

  const reasons = signal.reasons || [];
  if (reasons.length) {
    const why = el("ul", "call-reasons");
    reasons.forEach((reason) => why.appendChild(el("li", null, reason.sentence)));
    card.appendChild(why);
  }

  return card;
}

function callsPanel(run) {
  const signals = run.signals || [];
  const panel = el("section", "a-panel");
  panel.appendChild(el("h2", null, "Today"));

  if (!signals.length) {
    panel.appendChild(el("p", "empty", "No signals in this run."));
    return panel;
  }

  const grid = el("div", "calls");
  signals.forEach((signal) => grid.appendChild(signalCard(signal)));
  panel.appendChild(grid);

  panel.appendChild(el("p", "note",
    "The reasons describe what moved this model's answer today. They explain "
    + "the decision; they are not evidence the decision is right, and they are "
    + "worth reading only when the gate above is open."));

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

    const meta = el("p", "analysis-meta",
      `Model ${run.run_id}, ${run.horizon}-day horizon, graded on `
      + `${run.dataset?.test?.from || "?"} → ${run.dataset?.test?.to || "?"}.`);
    host.appendChild(meta);

    [trustPanel(run), learntPanel(run), callsPanel(run)]
      .filter(Boolean)
      .forEach((panel) => host.appendChild(panel));

  } catch (error) {
    host.replaceChildren(el("p", "error", `Could not load: ${error.message}`));
  }
}

refresh();
setInterval(refresh, POLL_MS);
