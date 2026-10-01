// The desk: what it said unprompted, and what it was asked.
//
// Built as a transcript rather than a chat session, because that is what the
// rest of this project is. Lines are appended on disk and dated, so what it
// said last Tuesday is still here and can be held against what the stores
// said at the time -- which is the only way an opinion is worth anything
// later. Nothing is edited and nothing scrolls away.
//
// Newest first, deliberately. A busy person opening this at the end of the
// day wants the last thing it noticed, not to scroll through the morning.

import {
  el, count, getJSON, poll, sectionHead, tag,
} from "ui";

const POLL_MS = 20000;

const KINDS = {
  note: { label: "noticed", tone: "forward" },
  question: { label: "you asked", tone: "quiet" },
  answer: { label: "answered", tone: "cool" },
  unanswered: { label: "no model", tone: "quiet" },
  quiet: { label: "nothing new", tone: "quiet" },
};

function when(stamp) {
  if (!stamp) return "";
  const at = new Date(stamp);
  if (Number.isNaN(at.getTime())) return stamp;
  const mins = Math.round((Date.now() - at.getTime()) / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  if (mins < 60 * 24) return `${Math.round(mins / 60)}h ago`;
  return at.toISOString().slice(0, 10);
}

// --- the transcript -----------------------------------------------------------

function line(entry) {
  const kind = KINDS[entry.kind] || { label: entry.kind, tone: "quiet" };
  const item = el("article", `said is-${entry.kind}`);

  const head = el("div", "said-head");
  head.appendChild(tag(kind.label, kind.tone));
  head.appendChild(el("span", "said-when", when(entry.at)));
  if (entry.model) head.appendChild(el("span", "said-model", entry.model));
  item.appendChild(head);

  if (entry.text) item.appendChild(el("p", "said-text", entry.text));

  // Why it spoke. The unprompted lines are the ones worth being able to
  // interrogate later -- "the archive changed" is a claim, and the digest
  // beside it says which findings it was about.
  if (entry.because) item.appendChild(el("p", "said-why", entry.because));

  const sources = entry.sources || [];
  if (sources.length) {
    const list = el("ul", "sources");
    sources.forEach((source) => {
      const row = el("li");
      if (source.url) {
        const link = el("a", null, source.title);
        link.href = source.url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        row.appendChild(link);
      } else {
        row.appendChild(el("span", null, source.title));
      }
      row.appendChild(el("span", "sources-meta", source.provider || "unknown"));
      list.appendChild(row);
    });
    item.appendChild(list);
  }

  return item;
}

// --- asking it something -------------------------------------------------------

function asker(canSpeak) {
  const form = el("form", "ask");
  const field = el("input", "ask-field");
  field.type = "text";
  field.name = "question";
  field.autocomplete = "off";
  field.placeholder = canSpeak
    ? "Ask about what is stored — a name, the archive, what it noticed"
    : "No model is answering, but the question is still recorded";

  const button = el("button", "btn", "Ask");
  button.type = "submit";

  form.appendChild(field);
  form.appendChild(button);

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const question = field.value.trim();
    if (!question) return;

    field.disabled = true;
    button.disabled = true;
    button.textContent = "Thinking…";

    try {
      await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      field.value = "";
    } finally {
      field.disabled = false;
      button.disabled = false;
      button.textContent = "Ask";
      field.focus();
      refresh();
    }
  });

  return form;
}

// --- the page ------------------------------------------------------------------

function describe(backend) {
  if (!backend || backend.kind === "none") {
    return {
      heading: "Nothing is answering yet",
      blurb: "Run a model on this machine — `ollama serve` and a 7-9B at "
        + "Q4 fits an 8GB card — or set an API key. Until then the "
        + "transcript records questions and the rest of the project carries "
        + "on without it.",
      can: false,
    };
  }
  const where = backend.kind === "local"
    ? `${backend.model}, running on this machine, free`
    : `${backend.model}, hosted, billed per call`;
  return {
    heading: "It speaks when the archive changes",
    blurb: `${where}. It is handed the measured findings and nothing else, so `
      + `it can restate them but cannot add a number to them. A cycle where `
      + `nothing moved writes nothing.`,
    can: true,
  };
}

async function refresh() {
  const host = document.getElementById("chat");
  const body = await getJSON("/api/chat");

  if (!body) {
    host.replaceChildren(el("p", "error", "Could not reach the server."));
    return;
  }

  const state = describe(body.backend);
  const parts = [];

  const head = el("section", "verdict");
  head.appendChild(el("div", "verdict-kicker", "the desk"));
  head.appendChild(el("h1", "verdict-word", state.heading));
  head.appendChild(el("p", "verdict-why", state.blurb));
  parts.push(head);

  const asking = el("section");
  asking.appendChild(asker(state.can));
  parts.push(asking);

  const feed = (body.feed || []).filter((entry) => entry.text);
  const transcript = el("section");
  transcript.appendChild(sectionHead(
    "transcript",
    feed.length
      ? `${count(feed.length)} line${feed.length === 1 ? "" : "s"}, newest first`
      : "Nothing said yet",
    "Appended and never edited, so what it believed last week can be checked "
    + "against what the stores held at the time."));

  if (!feed.length) {
    transcript.appendChild(el("p", "empty",
      state.can
        ? "It has not spoken yet. It writes a line when the scheduler finds "
          + "the archive has changed — or you can ask it something."
        : "Nothing has been asked and no model is running."));
  } else {
    const list = el("div", "transcript");
    feed.forEach((entry) => list.appendChild(line(entry)));
    transcript.appendChild(list);
  }
  parts.push(transcript);

  host.replaceChildren(...parts);
}

poll(refresh, POLL_MS);
