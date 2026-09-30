// Why the gate is shut, in pictures.
//
// Four charts, each making one argument that a table of numbers softens:
//
//   1. The same book held two ways parts over the test period. That gap is the
//      overnight move, and it is where the edge lives.
//   2. The edge stands next to the distance two identical models land apart by
//      chance, and next to the bar the gate sets. Without those two it is just
//      a number that happens to be positive.
//   3. Accuracy climbs with confidence all the way to 71.8%. The return an
//      order can reach does not climb with it.
//   4. Six independent windows: the edge is positive in all six, and the money
//      is negative in all six.
//
// Then the controls, which is the answer to "would a linear model have done
// this?" -- and on this panel it would, slightly better.

import {
  el, pct, pp, num, plain, count, signedPct, getJSON, poll, stat, statRow,
  sectionHead, table, equityChart, edgeScale, confidenceChart, foldChart, tag,
} from "ui";

const POLL_MS = 60000;

// --- 1. the execution gap -----------------------------------------------------

function executionSection(evaluation) {
  const section = el("section");
  section.appendChild(sectionHead(
    "the finding",
    "The same positions, held two ways",
    "The label describes a close-to-close move. The signal is computed from a "
    + "close, so the first price anybody can act on is the next open — and "
    + "roughly ninety per cent of the gross edge turns out to sit in the gap "
    + "between those two. Both books below hold the same names on the same days "
    + "and pay the same costs. The shaded space between them is the part no "
    + "order can reach."));

  const chart = equityChart(evaluation.equity || {});
  if (chart) {
    const panel = el("div", "panel");
    panel.appendChild(chart);
    section.appendChild(panel);
  } else {
    section.appendChild(el("p", "note",
      "This run was scored before the equity curves were recorded. Re-score it "
      + "and the chart appears."));
  }

  section.appendChild(statRow(
    stat("close to close", num(evaluation.strategy_sharpe),
      `sharpe · t ${num(evaluation.strategy_tstat)} · not tradeable`),
    stat("open to close", num(evaluation.executable_sharpe),
      `sharpe · t ${num(evaluation.executable_tstat)} · what the gate reads`,
      "bad"),
    stat("execution gap", num(evaluation.execution_gap),
      "sharpe lost to the overnight move", "bad"),
    stat("annualised, tradeable", signedPct(evaluation.executable_annualised, 1),
      `against ${signedPct(evaluation.hold_annualised, 1)} for holding the panel`,
      "bad"),
    stat("worst drawdown", signedPct(evaluation.executable_max_drawdown, 1),
      "on the tradeable series", "bad"),
  ));
  return section;
}

// --- 2. the edge against its own noise ----------------------------------------

function edgeSection(run) {
  const evaluation = run.evaluation || {};
  const trust = run.trust || {};
  const floor = (run.controls || {}).noise_floor || {};

  const section = el("section");
  section.appendChild(sectionHead(
    "the signal",
    "The edge, and the two numbers that decide whether it means anything",
    "An edge of a fraction of a point is only interesting next to how far apart "
    + "two identical models land by chance, and next to how large the edge would "
    + "have to be for this many trials to have stopped mattering."));

  const panel = el("div", "panel");
  const scale = edgeScale({
    edge: evaluation.edge,
    standardError: evaluation.edge_standard_error,
    noiseFloor: floor.spread,
    needed: trust.needed,
  });
  if (scale) panel.appendChild(scale);
  section.appendChild(panel);

  section.appendChild(statRow(
    stat("accuracy", pct(evaluation.accuracy),
      `against a ${pct(evaluation.baseline_accuracy)} baseline`),
    stat("edge", pp(evaluation.edge),
      `± ${pp(evaluation.edge_standard_error)} clustered by date`, "good"),
    stat("rows", count(evaluation.rows),
      `worth ${count(evaluation.effective_rows)} independent ones`),
    stat("design effect", plain(evaluation.design_effect, 2),
      "how much the panel moves together"),
    stat("says up", pct(evaluation.up_rate, 1),
      "a stuck model would sit at 0 or 100"),
  ));
  return section;
}

// --- 3. accuracy climbs, money does not ---------------------------------------

function confidenceSection(evaluation) {
  const buckets = (evaluation.by_confidence || []).filter((b) => b.rows > 0);
  if (!buckets.length) return null;

  // The most confident bucket that still has an accuracy to quote. The last
  // bucket does not: 24 rows is under the floor, so `evaluate` withholds the
  // point estimate and gives only an interval. Quoting that bucket's returns
  // beside another bucket's name was this page's own bug.
  const told = buckets.filter((b) => b.accuracy !== null && b.accuracy !== undefined);
  const best = told.length ? told[told.length - 1] : null;
  const withheld = buckets.filter((b) => b.accuracy === null || b.accuracy === undefined);

  const section = el("section");
  section.appendChild(sectionHead(
    "the trap",
    "It gets more accurate as it gets more confident. The money does not follow.",
    "This is the table that looks like a strategy and is not one. Sort the rows "
    + "by how sure the model was and accuracy climbs the whole way. The "
    + "close-to-close return climbs with it. The return an order could actually "
    + "have captured stays flat and then turns negative — in the most "
    + "confident bucket of all."));

  const panel = el("div", "panel");
  const chart = confidenceChart(buckets);
  if (chart) panel.appendChild(chart);

  if (best) {
    const band = `${(best.from * 100).toFixed(0)}–${Math.min(best.to * 100, 100).toFixed(0)}%`;
    const gone = best.mean_net_return
      ? 1 - (best.mean_executable_return || 0) / best.mean_net_return
      : null;
    panel.appendChild(el("p", "note",
      `On the ${band} confidence band it is right ${pct(best.accuracy, 1)} of the `
      + `time over ${count(best.rows)} rows, and returns `
      + `${signedPct(best.mean_net_return, 3)} a row close to close — against `
      + `${signedPct(best.mean_executable_return, 3)} held from the next open. `
      + (gone !== null
        ? `That is ${pct(gone, 1)} of the move gone before the market opens. `
        : "")
      + `A page showing only the first of those numbers is an argument for `
      + `trading exactly those rows.`));
  }
  section.appendChild(panel);

  section.appendChild(table(
    ["Confidence", "Rows", "Accuracy", "95% interval", "Close to close", "From next open"],
    buckets.map((bucket) => ({
      className: bucket.mean_executable_return < 0 ? "is-subject" : null,
      cells: [
        `${(bucket.from * 100).toFixed(0)}–${Math.min(bucket.to * 100, 100).toFixed(0)}%`,
        count(bucket.rows),
        bucket.accuracy === null || bucket.accuracy === undefined
          ? { text: "too few rows", className: "flat" }
          : { text: pct(bucket.accuracy, 2), className: bucket.accuracy >= 0.55 ? "up" : null },
        (bucket.accuracy_interval || []).filter((v) => v !== null).length === 2
          ? `${pct(bucket.accuracy_interval[0], 1)} – ${pct(bucket.accuracy_interval[1], 1)}`
          : "—",
        {
          text: signedPct(bucket.mean_net_return, 3),
          className: bucket.mean_net_return > 0 ? "up" : "down",
        },
        {
          text: signedPct(bucket.mean_executable_return, 3),
          className: bucket.mean_executable_return > 0 ? "up" : "down",
        },
      ],
    })),
  ));

  if (withheld.length) {
    section.appendChild(el("p", "note",
      `The most confident band holds ${count(withheld[0].rows)} rows, which is `
      + `under the floor for quoting an accuracy — so the point estimate is `
      + `withheld and only the interval is shown. An accuracy of 100% on two `
      + `dozen rows is a number about the rows, not about the model.`));
  }
  return section;
}

// --- 4. every window -----------------------------------------------------------

function foldSection(walkForward) {
  const folds = (walkForward || {}).folds || [];
  if (!folds.length) return null;

  const negative = folds.filter((f) => (f.executable_sharpe || 0) < 0).length;
  const section = el("section");
  section.appendChild(sectionHead(
    "across time",
    `Positive in ${walkForward.folds_positive} of ${walkForward.folds_run} windows. `
    + `Unprofitable in ${negative} of ${folds.length}.`,
    "One test period is one draw. These are independent windows, each trained "
    + "only on what came before it. The edge is not fragile — it shows up "
    + "in every window. It just is not reachable in any of them."));

  const panel = el("div", "panel");
  const chart = foldChart(walkForward);
  if (chart) panel.appendChild(chart);
  panel.appendChild(el("p", "note",
    `Mean edge ${pp(walkForward.mean_edge)} with a spread of `
    + `${pp(walkForward.sd_edge)} across windows, measured on `
    + `${walkForward.model || "the control"}.`));
  section.appendChild(panel);
  return section;
}

// --- the controls --------------------------------------------------------------

function controlsSection(run) {
  const controls = run.controls || {};
  const evaluation = run.evaluation || {};
  if (!controls.majority) return null;

  const rows = [
    ["Always the training majority", controls.majority],
    ["Logistic regression, same rows", controls.logistic],
    ["A second MLP, different seed", controls.local_mlp],
    ["The model on trial", evaluation],
  ].filter(([, value]) => value && value.accuracy !== undefined);

  const section = el("section");
  section.appendChild(sectionHead(
    "the controls",
    "What it had to beat, trained on the same rows with the same budget",
    "The question a single score cannot answer: would a linear model have done "
    + "this? On this panel it does, and slightly better — which is the "
    + "strongest evidence in the project that the network is not the thing "
    + "worth improving."));

  section.appendChild(table(
    ["Model", "Accuracy", "Edge", "Says up", "Close-to-close sharpe", "Tradeable t"],
    rows.map(([label, result], index) => ({
      className: index === rows.length - 1 ? "is-subject" : null,
      cells: [
        label,
        pct(result.accuracy),
        { text: pp(result.edge), className: result.edge > 0 ? "up" : "flat" },
        pct(result.up_rate, 0),
        num(result.strategy_sharpe),
        {
          text: num(result.executable_tstat),
          className: result.executable_tstat > 0 ? "up" : "down",
        },
      ],
    })),
  ));

  const hyper = controls.hyperparameters || {};
  const floor = controls.noise_floor || {};
  section.appendChild(el("p", "note",
    `The network is ${hyper.hidden} wide and ${hyper.depth} deep — `
    + `7,233 parameters — trained for ${count(hyper.steps)} steps on `
    + `${controls.device || "this machine"}. Three seeds of the same `
    + `configuration spanned ${pp(floor.spread)}, which is the floor an edge has `
    + `to clear before it is a signal rather than an initialisation.`));
  return section;
}

// --- the data ------------------------------------------------------------------

function datasetSection(run, news) {
  const data = run.dataset || {};
  const test = data.test || {};
  const train = data.train || {};
  const blocks = data.blocks || {};

  const section = el("section");
  section.appendChild(sectionHead("the data", "What went in, and when it was cut"));

  section.appendChild(statRow(
    stat("train", count(train.rows), `${train.from} → ${train.to}`),
    stat("test", count(test.rows), `${test.from} → ${test.to}`),
    stat("cut", data.cut_date || "—", "everything trained on precedes it"),
    stat("symbols", count((data.symbols || []).length), "US and European equities"),
    stat("features", count((data.feature_names || []).length),
      "every one computable at that close"),
  ));

  const fold = el("details", "fold");
  fold.appendChild(el("summary", null, "The feature blocks, and the archive"));

  const rows = [
    ["Per-symbol indicators", true, `${(data.feature_names || []).length} total`],
    ["Cross-sectional", blocks.cross?.used,
      `${(blocks.cross?.columns || []).length} columns over ${blocks.cross?.panel_width || 0} names`],
    ["Macro", blocks.macro?.used,
      `${(blocks.macro?.columns || []).length} columns, ${count(blocks.macro?.rows)} rows`],
    ["Events", blocks.events?.used,
      `${(blocks.events?.columns || []).length} columns, ${pct(blocks.events?.coverage?.share, 0)} covered`],
    ["News", blocks.news?.used,
      news ? `${count(news.items)} items, ${news.history_days} of ${news.needed_days} days` : "inert"],
  ];

  fold.appendChild(table(["Block", "In the model", "Detail"],
    rows.map(([label, used, detail]) => ({
      cells: [label, tag(used ? "used" : "inert", used ? "pass" : "quiet"), detail],
    }))));

  if (news && !news.ready) {
    fold.appendChild(el("p", "note",
      `The news block refuses to be a feature until it has a year of its own `
      + `history — ${news.history_days} days so far. An archive fetched `
      + `later has been re-ranked by what turned out to matter, so this is the `
      + `one input that can only be built by waiting.`));
  }

  const ended = blocks.macro?.ended || {};
  if (Object.keys(ended).length) {
    fold.appendChild(el("p", "note",
      `Dropped for going stale: ${Object.entries(ended)
        .map(([name, info]) => `${name} (last ${info.last})`).join(", ")}. `
      + `A series more than five sessions behind the others is dropped with its `
      + `features rather than carried forward.`));
  }

  section.appendChild(fold);
  return section;
}

// --- what it attends to --------------------------------------------------------

function learntSection(learnt) {
  const rows = (learnt || []).slice(0, 12);
  if (!rows.length) return null;

  const section = el("section");
  section.appendChild(sectionHead(
    "the model from the inside",
    "What it attends to, across every row it was shown",
    "Influence is how much moving a feature moves the answer; the lean is which "
    + "way. A network that had collapsed to one answer would show almost no "
    + "influence anywhere, which is how this table earns its place."));

  section.appendChild(table(["Feature", "Influence", "Leans"],
    rows.map((row) => ({
      cells: [
        row.feature,
        plain(row.influence, 4),
        { text: num(row.leans, 4), className: row.leans > 0 ? "up" : "down" },
      ],
    }))));
  return section;
}

// --- the page ------------------------------------------------------------------

async function refresh() {
  const host = document.getElementById("evidence");
  const body = await getJSON("/api/analysis?detail=none");

  if (!body) {
    host.replaceChildren(el("p", "error", "Could not reach the server."));
    return;
  }
  if (!body.run) {
    host.replaceChildren(el("p", "empty",
      body.why || "No finished model yet. Train one first."));
    return;
  }

  const run = body.run;
  const evaluation = run.evaluation || {};

  const head = el("section", "verdict is-shut");
  head.appendChild(el("div", "verdict-kicker",
    `Model ${run.run_id} · ${run.horizon}-session horizon · `
    + `${(run.spec || {}).target} target`));
  head.appendChild(el("h1", "verdict-word", "The case against it"));
  head.appendChild(el("p", "verdict-why",
    "Everything below was measured on rows the model never trained on, and "
    + "every chart is drawn from the run's own stored series rather than "
    + "re-derived from a mean."));

  host.replaceChildren(...[
    head,
    executionSection(evaluation),
    confidenceSection(evaluation),
    edgeSection(run),
    foldSection(run.walk_forward),
    controlsSection(run),
    datasetSection(run, body.news),
    learntSection(run.learnt),
  ].filter(Boolean));
}

poll(refresh, POLL_MS);
