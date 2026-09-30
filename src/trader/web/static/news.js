// The archive, read aloud.
//
// The model cannot use this for a year, so the risk on this page is not being
// wrong -- it is being mistaken for evidence. Three things guard against that:
//
//   * The reading leads, and the caveat travels with it rather than sitting in
//     a footnote nobody reaches.
//   * Every claim carries the count it was computed from, and every suggestion
//     carries the headlines that prompted it, so it can be dismissed in ten
//     seconds if it is noise.
//   * Where the archive is too young to support a comparison, the page says
//     so in place of the number instead of printing one anyway.

import {
  el, pct, signedPct, count, plain, getJSON, poll, stat, statRow,
  sectionHead, table, tag, sign,
} from "ui";

const POLL_MS = 60000;

// --- what it makes of today ---------------------------------------------------

const TONE = {
  suggestion: "forward",
  "market-gap": "forward",
  name: "cool",
  market: "quiet",
  "suggestion-weak": "quiet",
  quiet: "quiet",
};

function reading(body) {
  const lines = body.reading || [];
  const section = el("section");

  const panel = el("section", "verdict");
  panel.appendChild(el("div", "verdict-kicker", "the archive"));
  panel.appendChild(el("h1", "verdict-word", "What it is reading"));

  const list = el("ul", "ledger");
  lines.forEach((line) => {
    const item = el("li", `is-${line.kind}`);
    item.appendChild(el("span", "ledger-when", line.kind.replace("-", " ")));
    item.appendChild(el("span", "ledger-text", line.text));
    list.appendChild(item);
  });
  panel.appendChild(list);
  panel.appendChild(el("p", "note", body.caveat));
  section.appendChild(panel);
  return section;
}

// --- where the attention is ---------------------------------------------------

function marketSection(body) {
  const rows = (body.markets || []).filter((row) => row.articles > 0);
  if (!rows.length) return null;

  const section = el("section");
  section.appendChild(sectionHead(
    "coverage",
    "Where the attention is, by market",
    "A ticker suffix is a market, so this groups itself and stays right when "
    + "the universe changes. Counted per tracked name, because a market with "
    + "forty names is not louder than one with four just for being larger."));

  section.appendChild(table(
    ["Market", "Tracked", "Articles", "Per name", "Vs its normal"],
    rows.map((row) => ({
      cells: [
        row.market,
        count(row.tracked),
        count(row.articles),
        plain(row.per_name, 1),
        row.times === null || row.times === undefined
          ? { text: "not enough history", className: "flat" }
          : { text: `${plain(row.times, 1)}×`,
              className: row.times >= 1.5 ? "up" : "flat" },
      ],
    })),
  ));
  return section;
}

// --- the names, joined to their price -----------------------------------------

function loudSection(body) {
  const rows = body.loudest || [];
  if (!rows.length) return null;

  const section = el("section");
  section.appendChild(sectionHead(
    "names",
    "The loudest names this week, and what the price did",
    "Coverage and price are two stores that had never been joined. A name in "
    + "three times its usual coverage whose price did nothing is a different "
    + "thing from one that moved, and neither is visible from either store "
    + "alone."));

  const panel = el("div", "panel");
  rows.forEach((row) => {
    const item = el("div", "attn");
    const head = el("div", "attn-head");
    head.appendChild(el("span", "attn-symbol", row.symbol));
    head.appendChild(el("span", "attn-market", row.market));

    head.appendChild(el("span", "attn-count",
      `${count(row.articles)} article${row.articles === 1 ? "" : "s"}`));
    if (row.times) {
      head.appendChild(tag(`${plain(row.times, 1)}× normal`, "forward"));
    } else {
      head.appendChild(tag("no baseline yet", "quiet"));
    }
    if (row.move !== null && row.move !== undefined) {
      head.appendChild(el("span", `attn-move ${sign(row.move)}`,
        signedPct(row.move, 1)));
    }
    item.appendChild(head);

    (row.headlines || []).forEach((title) => {
      item.appendChild(el("p", "attn-headline", title));
    });
    panel.appendChild(item);
  });
  section.appendChild(panel);
  return section;
}

// --- what it thinks is missing ------------------------------------------------

function evidence(headlines) {
  const list = el("ul", "sources");
  (headlines || []).forEach((item) => {
    const entry = el("li");
    if (item.url) {
      const link = el("a", null, item.title);
      link.href = item.url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      entry.appendChild(link);
    } else {
      entry.appendChild(el("span", null, item.title));
    }
    entry.appendChild(el("span", "sources-meta",
      `${item.provider || "unknown"} · ${item.seen}`));
    list.appendChild(entry);
  });
  return list;
}

function suggestionSection(body) {
  const rows = body.suggestions || [];
  const strong = rows.filter((row) => row.corroborated);
  const weak = rows.filter((row) => !row.corroborated);

  const section = el("section");
  section.appendChild(sectionHead(
    "names it is missing",
    strong.length
      ? `${strong.length} name${strong.length === 1 ? "" : "s"} the archive keeps naming that nothing tracks`
      : "Nothing has come up in enough separate contexts to suggest",
    "Roughly a quarter of stored headlines name a company outright, because "
    + "the wires write them that way. Those are extracted, mapped onto this "
    + "project's ticker convention, and the ones that resolve to something "
    + "untracked are ranked by how many separate companies' coverage they turn "
    + "up in — a name seen in only one is usually that company's other "
    + "listing, not a new one."));

  if (!rows.length) {
    section.appendChild(el("p", "empty",
      "No untracked name has cleared the bar. With three weeks of archive "
      + "that is the expected answer."));
    return section;
  }

  const render = (row) => {
    const panel = el("div", "panel");
    const head = el("div", "attn-head");
    head.appendChild(el("span", "attn-symbol", row.symbol));
    head.appendChild(el("span", "attn-market",
      `${row.market} · ${row.exchange}`));
    head.appendChild(tag(
      row.corroborated ? `${row.contexts} contexts` : "one context only",
      row.corroborated ? "forward" : "quiet"));
    panel.appendChild(head);

    panel.appendChild(el("p", "note",
      `Named in ${count(row.mentions)} stored items over ${row.days} days by `
      + `${row.providers} providers, in the coverage of `
      + `${row.alongside.join(", ")}. `
      + (row.corroborated
        ? `${row.tracked_in_market} name${row.tracked_in_market === 1 ? " is" : "s are"} `
          + `tracked in ${row.market}.`
        : "Seen only in that one company's coverage, which is usually what a "
          + "second listing of a name already tracked looks like.")));

    panel.appendChild(evidence(row.headlines));
    return panel;
  };

  strong.forEach((row) => section.appendChild(render(row)));

  if (weak.length) {
    const fold = el("details", "fold");
    fold.appendChild(el("summary", null,
      `Seen in one context only (${weak.length}) — probably second listings`));
    weak.forEach((row) => fold.appendChild(render(row)));
    section.appendChild(fold);
  }
  return section;
}

function marketGapSection(body) {
  const rows = body.unwatched_markets || [];
  if (!rows.length) return null;

  const section = el("section");
  section.appendChild(sectionHead(
    "markets it is missing",
    `${rows.length} market${rows.length === 1 ? "" : "s"} the archive names where nothing is tracked`,
    "The larger version of the same question, and the one only this page can "
    + "ask: the model cannot miss a market it was never given a column for."));

  rows.forEach((row) => {
    const panel = el("div", "panel");
    const head = el("div", "attn-head");
    head.appendChild(el("span", "attn-symbol", row.market));
    head.appendChild(tag("nothing tracked here", "forward"));
    panel.appendChild(head);
    panel.appendChild(el("p", "note",
      `${row.distinct_names} different compan`
      + `${row.distinct_names === 1 ? "y" : "ies"} named over ${row.days} days `
      + `by ${row.providers} provider${row.providers === 1 ? "" : "s"}: `
      + `${row.names.join(", ")}.`));
    panel.appendChild(evidence(row.headlines));
    section.appendChild(panel);
  });
  return section;
}

// --- the store itself ----------------------------------------------------------

function storeSection(body) {
  const state = body.readiness || {};
  const section = el("section");
  section.appendChild(sectionHead(
    "the store",
    "What has been collected, and how far it is from being usable",
    "This is the one input that cannot be bought later: an archive fetched "
    + "next year has been re-ranked by what turned out to matter, and its "
    + "timestamps are often ingestion times rather than publication ones. So "
    + "the only honest version is built forwards, which means waiting."));

  const share = state.needed_days
    ? Math.min(state.history_days / state.needed_days, 1) : 0;

  section.appendChild(statRow(
    stat("items", count(state.items), "titles, summaries and sources"),
    stat("history", `${count(state.history_days)}d`,
      `of ${count(state.needed_days)} before it may be a feature`),
    stat("progress", pct(share, 0), "towards being usable by the model"),
    stat("in the model", state.ready ? "yes" : "no",
      state.ready ? "the block is live" : "the block refuses zeros",
      state.ready ? "good" : null),
  ));
  return section;
}

// --- the page ------------------------------------------------------------------

async function refresh() {
  const host = document.getElementById("news");
  const body = await getJSON("/api/attention");

  if (!body) {
    host.replaceChildren(el("p", "error", "Could not reach the server."));
    return;
  }
  if (body.error) {
    host.replaceChildren(el("p", "error", `Could not read the archive: ${body.error}`));
    return;
  }

  host.replaceChildren(...[
    reading(body),
    suggestionSection(body),
    marketGapSection(body),
    loudSection(body),
    marketSection(body),
    storeSection(body),
  ].filter(Boolean));
}

document.getElementById("collect")?.addEventListener("click", async (event) => {
  const button = event.currentTarget;
  button.disabled = true;
  button.textContent = "Collecting…";
  try {
    await fetch("/api/news/collect", { method: "POST" });
  } finally {
    setTimeout(() => {
      button.disabled = false;
      button.textContent = "Collect now";
      refresh();
    }, 4000);
  }
});

poll(refresh, POLL_MS);
