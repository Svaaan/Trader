// Shared drawing: elements, number formats, and four charts.
//
// Built with createElement and createElementNS throughout, never innerHTML.
// Symbol names and error strings arrive from yfinance and from news providers,
// and a page that puts those through innerHTML is a page that will one day
// render whatever a data provider decided to put in a field.
//
// The charts are hand-drawn SVG rather than a library. Four charts do not pay
// for a charting dependency, and every one of them here is making a specific
// argument that a generic chart would soften: the gap between two equity
// curves, the distance from an edge to its own noise floor, accuracy rising
// while money does not. Drawing them by hand keeps the argument explicit.

// --- elements -----------------------------------------------------------------

export function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

export function frag(...children) {
  const f = document.createDocumentFragment();
  children.filter(Boolean).forEach((c) => f.appendChild(c));
  return f;
}

const NS = "http://www.w3.org/2000/svg";

export function s(tag, attrs = {}, text) {
  const node = document.createElementNS(NS, tag);
  Object.entries(attrs).forEach(([k, v]) => {
    if (v !== undefined && v !== null) node.setAttribute(k, String(v));
  });
  if (text !== undefined) node.textContent = String(text);
  return node;
}

// --- numbers ------------------------------------------------------------------

const missing = (v) => v === null || v === undefined || Number.isNaN(v);

export const pct = (v, d = 2) => (missing(v) ? "—" : `${(v * 100).toFixed(d)}%`);

export const signedPct = (v, d = 2) =>
  missing(v) ? "—" : `${v >= 0 ? "+" : ""}${(v * 100).toFixed(d)}%`;

// Percentage points, for a difference between two percentages. "+0.74pp" and
// "+0.74%" are different claims and the page was making the wrong one.
export const pp = (v, d = 2) =>
  missing(v) ? "—" : `${v >= 0 ? "+" : ""}${(v * 100).toFixed(d)}pp`;

// Plain number. A t-statistic is not a percentage; it was being rendered as
// "-318.50%" before this existed.
export const num = (v, d = 2) => (missing(v) ? "—" : `${v >= 0 ? "+" : ""}${v.toFixed(d)}`);

export const plain = (v, d = 2) => (missing(v) ? "—" : v.toFixed(d));

export const money = (v, d = 2) => (missing(v) ? "—" : `$${Number(v).toFixed(d)}`);

export const count = (v) => (missing(v) ? "—" : Number(v).toLocaleString("en-US"));

export const sign = (v) => (missing(v) ? "flat" : v > 0 ? "up" : v < 0 ? "down" : "flat");

// --- small components ---------------------------------------------------------

export function stat(label, value, hint, tone) {
  const box = el("div", "stat");
  box.appendChild(el("div", "stat-label", label));
  box.appendChild(el("div", `stat-value${tone ? ` is-${tone}` : ""}`, value));
  if (hint) box.appendChild(el("div", "stat-hint", hint));
  return box;
}

export function statRow(...stats) {
  const row = el("div", "stats");
  stats.filter(Boolean).forEach((x) => row.appendChild(x));
  return row;
}

export function tag(text, tone = "quiet") {
  return el("span", `tag is-${tone}`, text);
}

export function sectionHead(eyebrow, title, blurb) {
  const head = el("div", "section-head");
  if (eyebrow) head.appendChild(el("div", "eyebrow", eyebrow));
  head.appendChild(el("h2", null, title));
  if (blurb) head.appendChild(el("p", null, blurb));
  return head;
}

export function legend(keys) {
  const box = el("div", "legend");
  keys.forEach(({ color, label, dashed }) => {
    const key = el("div", "legend-key");
    const swatch = el("span", "legend-swatch");
    swatch.style.background = dashed
      ? `repeating-linear-gradient(90deg, ${color} 0 4px, transparent 4px 7px)`
      : color;
    key.appendChild(swatch);
    key.appendChild(el("span", null, label));
    box.appendChild(key);
  });
  return box;
}

export function table(headers, rows) {
  const wrap = el("div", "table-scroll");
  const t = el("table");
  const thead = el("thead");
  const hr = el("tr");
  headers.forEach((h) => hr.appendChild(el("th", null, h)));
  thead.appendChild(hr);
  t.appendChild(thead);

  const tbody = el("tbody");
  rows.forEach((cells) => {
    const tr = el("tr", cells.className);
    (cells.cells || cells).forEach((cell) => {
      if (cell && typeof cell === "object" && !(cell instanceof Node)) {
        tr.appendChild(el("td", cell.className, cell.text));
      } else if (cell instanceof Node) {
        const td = el("td");
        td.appendChild(cell);
        tr.appendChild(td);
      } else {
        tr.appendChild(el("td", null, cell));
      }
    });
    tbody.appendChild(tr);
  });
  t.appendChild(tbody);
  wrap.appendChild(t);
  return wrap;
}

// --- the gate -----------------------------------------------------------------

export function pips(checks) {
  const row = el("div", "pips");
  (checks || []).forEach((check) => {
    const pip = el("div", `pip${check.passed ? "" : " is-fail"}`);
    pip.title = `${check.name}: ${check.detail}`;
    row.appendChild(pip);
  });
  const cleared = (checks || []).filter((c) => c.passed).length;
  const total = (checks || []).length;
  row.appendChild(el("div", "pip-count", `${cleared} of ${total} cleared`));
  return row;
}

const HURDLE_NAMES = {
  enough_days: "Enough distinct days",
  enough_effective_rows: "Enough independent rows",
  model_varies: "It gives more than one answer",
  beats_baseline: "It beats the baseline anyone had in advance",
  beats_chance: "The edge is larger than chance",
  beats_noise_floor: "The edge is larger than the seed spread",
  consistent_across_time: "It holds across walk-forward windows",
  makes_money: "The money is distinguishable from luck",
};

export function hurdleName(name) {
  return HURDLE_NAMES[name] || String(name).replace(/_/g, " ");
}

export function gateRail(checks) {
  const rail = el("div", "gate-rail");
  (checks || []).forEach((check) => {
    const row = el("div", `hurdle${check.passed ? "" : " is-fail"}`);
    row.appendChild(el("div", "hurdle-mark", check.passed ? "✓" : "✕"));
    const body = el("div");
    body.appendChild(el("div", "hurdle-name", hurdleName(check.name)));
    body.appendChild(el("p", "hurdle-detail", check.detail));
    row.appendChild(body);
    rail.appendChild(row);
  });
  return rail;
}

// --- chart scaffolding --------------------------------------------------------

function axes(root, { x0, x1, y0, y1, ticks, label }) {
  ticks.forEach(({ at, text }) => {
    root.appendChild(s("line", {
      class: Math.abs(at - y1) < 0.5 ? "axis-line" : "grid-line",
      x1: x0, x2: x1, y1: at, y2: at, "stroke-width": 1,
    }));
    root.appendChild(s("text", {
      x: x0 - 8, y: at + 3.5, "text-anchor": "end", "font-size": 10.5,
    }, text));
  });
  if (label) {
    root.appendChild(s("text", {
      x: x0 - 8, y: y0 - 12, "text-anchor": "end", "font-size": 10,
    }, label));
  }
}

function path(points, close) {
  return points.reduce((d, [x, y], i) =>
    d + `${i === 0 ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)}`, "")
    + (close || "");
}

// --- 1. the execution gap -----------------------------------------------------

// The chart this whole project earns. Two books, the same positions, the same
// costs -- one held close to close, which is what the label describes, and one
// held from the first open that exists after the signal does. They start at the
// same 1.0 and the space between them is the overnight move, which is where
// roughly ninety per cent of the gross edge turned out to live.
export function equityChart(equity, opts = {}) {
  const dates = equity.dates || [];
  if (dates.length < 2) return null;

  const W = 960;
  const H = opts.height || 320;
  const pad = { l: 52, r: 62, t: 18, b: 28 };
  const x0 = pad.l;
  const x1 = W - pad.r;
  const y0 = pad.t;
  const y1 = H - pad.b;

  const series = [
    { key: "hold", color: "#667085", label: "Buy and hold the panel", dashed: true },
    { key: "graded", color: "#56c2f5", label: "Close to close — what the label describes" },
    { key: "executable", color: "#ff6b6b", label: "Open to close — what an order can reach" },
  ].filter((entry) => (equity[entry.key] || []).length === dates.length);

  const all = series.flatMap((entry) => equity[entry.key]);
  const lo = Math.min(...all, 1);
  const hi = Math.max(...all, 1);
  const padding = (hi - lo) * 0.12 || 0.02;
  const min = lo - padding;
  const max = hi + padding;

  const sx = (i) => x0 + (i / (dates.length - 1)) * (x1 - x0);
  const sy = (v) => y1 - ((v - min) / (max - min)) * (y1 - y0);

  const root = s("svg", {
    class: "chart", viewBox: `0 0 ${W} ${H}`,
    role: "img",
    "aria-label": "Equity from 1.0 over the sealed test period, held close to "
      + "close and held from the next open",
  });

  // Gridlines at round equity levels, so a reader can see 1.0 and read off a loss.
  const step = (max - min) > 0.4 ? 0.1 : 0.05;
  const ticks = [];
  for (let v = Math.ceil(min / step) * step; v <= max; v += step) {
    ticks.push({ at: sy(v), text: `${v >= 1 ? "+" : ""}${((v - 1) * 100).toFixed(0)}%` });
  }
  axes(root, { x0, x1, y0, y1, ticks, label: "equity" });

  // 1.0 is the line that matters: above it the account made money.
  root.appendChild(s("line", {
    class: "zero-line", x1: x0, x2: x1, y1: sy(1), y2: sy(1), "stroke-width": 1,
  }));

  // The gap itself, shaded. This is the subject of the chart, not the lines.
  const graded = equity.graded || [];
  const reachable = equity.executable || [];
  if (graded.length === dates.length && reachable.length === dates.length) {
    const top = graded.map((v, i) => [sx(i), sy(v)]);
    const bottom = reachable.map((v, i) => [sx(i), sy(v)]).reverse();
    root.appendChild(s("path", {
      d: path(top.concat(bottom), "Z"),
      fill: "rgba(255, 107, 107, 0.10)", stroke: "none",
    }));
  }

  series.forEach((entry) => {
    const values = equity[entry.key];
    root.appendChild(s("path", {
      d: path(values.map((v, i) => [sx(i), sy(v)])),
      fill: "none", stroke: entry.color,
      "stroke-width": entry.key === "hold" ? 1.25 : 1.9,
      "stroke-dasharray": entry.dashed ? "3 3" : null,
      "stroke-linejoin": "round",
    }));

    // Where it ended, at the end of its own line, so no legend lookup is needed.
    const last = values[values.length - 1];
    root.appendChild(s("text", {
      x: x1 + 7, y: sy(last) + 3.5, "font-size": 11, fill: entry.color,
    }, `${last >= 1 ? "+" : ""}${((last - 1) * 100).toFixed(1)}%`));
  });

  // A few dates along the bottom; the exact days are not the point.
  [0, Math.floor(dates.length / 3), Math.floor((2 * dates.length) / 3),
    dates.length - 1].forEach((i, n, list) => {
    root.appendChild(s("text", {
      x: sx(i), y: H - 8, "font-size": 10.5,
      "text-anchor": n === 0 ? "start" : n === list.length - 1 ? "end" : "middle",
    }, dates[i].slice(0, 7)));
  });

  const wrap = el("div");
  wrap.appendChild(root);
  wrap.appendChild(legend(series.map(({ color, label, dashed }) => ({ color, label, dashed }))));
  return wrap;
}

// --- 2. where the edge sits against its own noise -----------------------------

// An edge means nothing without the two numbers beside it: how far apart two
// identical models land by chance, and how far the gate needs it to be. All
// three on one axis, because that is the comparison and a table hides it.
export function edgeScale({ edge, standardError, noiseFloor, needed }) {
  if (edge === undefined || edge === null) return null;

  const W = 960;
  const H = 108;
  const x0 = 24;
  const x1 = W - 24;
  const mid = 58;

  const span = Math.max(edge + (standardError || 0) * 2, needed || 0,
    noiseFloor || 0, 0.002) * 1.35;
  const sx = (v) => x0 + (v / span) * (x1 - x0);

  const root = s("svg", {
    class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": "The measured edge against the seed spread and the bar the gate sets",
  });

  root.appendChild(s("line", {
    class: "axis-line", x1: x0, x2: x1, y1: mid, y2: mid, "stroke-width": 1,
  }));

  // Everything below the noise floor is a seed, not a signal.
  if (noiseFloor) {
    root.appendChild(s("rect", {
      x: x0, y: mid - 14, width: sx(noiseFloor) - x0, height: 28,
      fill: "rgba(255, 255, 255, 0.05)",
    }));
    root.appendChild(s("text", {
      x: sx(noiseFloor) - 6, y: mid + 30, "font-size": 10.5, "text-anchor": "end",
    }, `seed spread ${(noiseFloor * 100).toFixed(2)}pp`));
  }

  // The bar, as a line you can see the edge standing next to.
  if (needed) {
    root.appendChild(s("line", {
      x1: sx(needed), x2: sx(needed), y1: mid - 20, y2: mid + 20,
      stroke: "#f5b942", "stroke-width": 1.5, "stroke-dasharray": "4 3",
    }));
    root.appendChild(s("text", {
      x: sx(needed), y: mid - 27, "font-size": 10.5, "text-anchor": "middle",
      fill: "#f5b942",
    }, `bar ${(needed * 100).toFixed(2)}pp`));
  }

  // The edge, with two standard errors either side of it.
  if (standardError) {
    root.appendChild(s("line", {
      x1: sx(Math.max(edge - standardError * 2, 0)), x2: sx(edge + standardError * 2),
      y1: mid, y2: mid, stroke: "#56c2f5", "stroke-width": 7,
      "stroke-linecap": "round", opacity: 0.32,
    }));
  }
  root.appendChild(s("circle", { cx: sx(edge), cy: mid, r: 6, fill: "#56c2f5" }));
  root.appendChild(s("text", {
    x: sx(edge), y: mid + 30, "font-size": 12, "text-anchor": "middle", fill: "#56c2f5",
  }, `edge ${(edge * 100).toFixed(2)}pp`));

  const wrap = el("div");
  wrap.appendChild(root);
  wrap.appendChild(el("p", "note",
    standardError
      ? `The bar is drawn with the edge's own spread — plus and minus two `
        + `standard errors, clustered by date, so a panel that moves together `
        + `is not counted as hundreds of independent verdicts.`
      : "No standard error was recorded for this run."));
  return wrap;
}

// --- 3. accuracy climbs, money does not ---------------------------------------

// The most damning table in the project, as a chart. Confidence rises left to
// right; accuracy rises with it and keeps rising; the close-to-close return
// rises with it too. The return an order can reach does not.
export function confidenceChart(buckets) {
  const rows = (buckets || []).filter((b) => b.rows > 0);
  if (!rows.length) return null;

  const W = 960;
  const H = 300;
  const pad = { l: 52, r: 52, t: 22, b: 58 };
  const x0 = pad.l;
  const x1 = W - pad.r;
  const y0 = pad.t;
  const y1 = H - pad.b;

  const returns = rows.flatMap((b) =>
    [b.mean_net_return, b.mean_executable_return].filter((v) => v !== null && v !== undefined));
  const reach = Math.max(...returns.map(Math.abs), 0.001) * 1.2;
  const ry = (v) => y1 - ((v + reach) / (2 * reach)) * (y1 - y0);

  const band = (x1 - x0) / rows.length;
  const root = s("svg", {
    class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": "Accuracy and mean return per confidence bucket, in both holding windows",
  });

  const ticks = [-reach, -reach / 2, 0, reach / 2, reach].map((v) => ({
    at: ry(v), text: `${v >= 0 ? "+" : ""}${(v * 100).toFixed(2)}%`,
  }));
  axes(root, { x0, x1, y0, y1, ticks, label: "return per row" });

  root.appendChild(s("line", {
    class: "zero-line", x1: x0, x2: x1, y1: ry(0), y2: ry(0), "stroke-width": 1,
  }));

  const bw = Math.min(band * 0.3, 46);

  rows.forEach((bucket, i) => {
    const centre = x0 + band * (i + 0.5);

    [{ key: "mean_net_return", color: "#56c2f5", offset: -bw - 3 },
     { key: "mean_executable_return", color: "#ff6b6b", offset: 3 }]
      .forEach(({ key, color, offset }) => {
        const v = bucket[key];
        if (v === null || v === undefined) return;
        const top = Math.min(ry(v), ry(0));
        root.appendChild(s("rect", {
          x: centre + offset, y: top, width: bw,
          height: Math.max(Math.abs(ry(v) - ry(0)), 1),
          fill: color, opacity: 0.85, rx: 2,
        }));
      });

    // The label under each band carries the thing the bars cannot: how often
    // it was right, and on how many rows.
    root.appendChild(s("text", {
      x: centre, y: H - 34, "font-size": 11, "text-anchor": "middle",
      fill: bucket.accuracy >= 0.55 ? "#3ddc97" : "#98a2b3",
    }, bucket.accuracy === null ? "—" : `${(bucket.accuracy * 100).toFixed(1)}% right`));

    root.appendChild(s("text", {
      x: centre, y: H - 19, "font-size": 10, "text-anchor": "middle",
    }, `${(bucket.from * 100).toFixed(0)}–${Math.min(bucket.to * 100, 100).toFixed(0)}% conf`));

    root.appendChild(s("text", {
      x: centre, y: H - 6, "font-size": 9.5, "text-anchor": "middle",
    }, `${count(bucket.rows)} rows`));
  });

  const wrap = el("div");
  wrap.appendChild(root);
  wrap.appendChild(legend([
    { color: "#56c2f5", label: "Close to close, net of costs" },
    { color: "#ff6b6b", label: "Held from the next open, net of costs" },
  ]));
  return wrap;
}

// --- 4. every window, both ways -----------------------------------------------

// Six independent windows. The edge is positive in all six and the money an
// order could reach is negative in all six, which is the result stated as
// plainly as it can be: it is not that the edge is fragile, it is that the edge
// is real and sits where nobody can take it.
export function foldChart(walkForward) {
  const folds = (walkForward || {}).folds || [];
  if (!folds.length) return null;

  const W = 960;
  const H = 260;
  const pad = { l: 52, r: 20, t: 26, b: 46 };
  const x0 = pad.l;
  const x1 = W - pad.r;
  const y0 = pad.t;
  const y1 = H - pad.b;
  const band = (x1 - x0) / folds.length;

  const reach = Math.max(...folds.map((f) => Math.abs(f.executable_sharpe || 0)),
    ...folds.map((f) => Math.abs(f.sharpe || 0)), 1) * 1.15;
  const sy = (v) => y1 - ((v + reach) / (2 * reach)) * (y1 - y0);

  const root = s("svg", {
    class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": "Sharpe per walk-forward window, graded and executable",
  });

  const ticks = [-reach, -reach / 2, 0, reach / 2, reach].map((v) => ({
    at: sy(v), text: v.toFixed(1),
  }));
  axes(root, { x0, x1, y0, y1, ticks, label: "sharpe" });
  root.appendChild(s("line", {
    class: "zero-line", x1: x0, x2: x1, y1: sy(0), y2: sy(0), "stroke-width": 1,
  }));

  const bw = Math.min(band * 0.32, 40);

  folds.forEach((fold, i) => {
    const centre = x0 + band * (i + 0.5);
    [{ v: fold.sharpe, color: "#56c2f5", offset: -bw - 3 },
     { v: fold.executable_sharpe, color: "#ff6b6b", offset: 3 }]
      .forEach(({ v, color, offset }) => {
        if (v === null || v === undefined) return;
        root.appendChild(s("rect", {
          x: centre + offset, y: Math.min(sy(v), sy(0)), width: bw,
          height: Math.max(Math.abs(sy(v) - sy(0)), 1), fill: color,
          opacity: 0.85, rx: 2,
        }));
      });

    root.appendChild(s("text", {
      x: centre, y: H - 26, "font-size": 10.5, "text-anchor": "middle", fill: "#3ddc97",
    }, `edge ${(fold.edge * 100).toFixed(2)}pp`));
    root.appendChild(s("text", {
      x: centre, y: H - 11, "font-size": 10, "text-anchor": "middle",
    }, (fold.test_from || "").slice(0, 7)));
  });

  const wrap = el("div");
  wrap.appendChild(root);
  wrap.appendChild(legend([
    { color: "#56c2f5", label: "Close-to-close Sharpe" },
    { color: "#ff6b6b", label: "Sharpe an order could reach" },
  ]));
  return wrap;
}

// --- polling ------------------------------------------------------------------

export async function getJSON(url) {
  try {
    const response = await fetch(url);
    if (!response.ok) return null;
    return await response.json();
  } catch (error) {
    return null;
  }
}

export function poll(fn, ms) {
  fn();
  setInterval(fn, ms);
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) fn();
  });
}
