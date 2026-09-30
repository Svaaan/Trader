"""What it noticed today, in its own words, written down before tomorrow.

Everything else here is built for a verdict that is months or years away: the
contest needs a hundred and twenty shared sessions, the news archive needs a
year, a promotion needs both. Meanwhile the system fetches a few hundred price
histories, a macro block, an earnings calendar and a growing pile of articles
every single day, and says almost none of it out loud.

So this is the other output: a short, dated note about what changed. It is not
a forecast and it is not a recommendation -- the gate decides those, and the
gate is shut. It is the part of the work that is useful immediately, because
noticing that a name it holds has lost the reason it was bought, or that a
market has gone quiet while another fills with articles, is worth knowing on
the day rather than in the summary of a backtest two years from now.

Two rules keep it honest:

**It only says what it measured.** Every line is computed from the stores --
the book record, the news archive, the run's own evaluation -- and carries the
number behind it. There is no language model here and nothing is generated.

**It is written forward and never edited.** Each note is appended with the date
it was written, so a week of them is a record of what the system believed at
the time, which is the only kind of opinion worth checking later.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os

import numpy as np

from . import holding as holding_mod
from . import news as news_mod

logger = logging.getLogger(__name__)

BRIEFING_DIR = os.environ.get(
    "TRADER_BRIEFING",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                 "data", "briefing"),
)
BRIEFING_FILE = "briefing.jsonl"

# How unusual a symbol's news flow has to be before it is worth a line. Three
# times its own recent average, not a fixed count, because a name that gets two
# articles a week and suddenly gets six is the interesting case -- and a name
# that always gets forty is not news at all.
NEWS_SPIKE = 3.0
NEWS_WINDOW_DAYS = 30
MIN_ITEMS_FOR_SPIKE = 3

# Nothing below this many sessions of shared record is worth reporting as a
# standing in the contest; it would read as a league table of noise.
MIN_CONTEST_SESSIONS = 20


def _path() -> str:
    return os.path.join(os.path.abspath(BRIEFING_DIR), BRIEFING_FILE)


def read(limit: int = 30) -> list:
    """The notes, newest first."""
    try:
        with open(_path(), encoding="utf-8") as handle:
            notes = [json.loads(line) for line in handle if line.strip()]
    except (OSError, ValueError):
        return []
    return list(reversed(notes))[:limit]


def _append(note: dict) -> dict:
    path = _path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(note, ensure_ascii=False) + "\n")
    return note


def news_spikes(symbols, *, now: dt.datetime | None = None) -> list:
    """Names the archive has suddenly filled up with, against their own normal.

    The archive is nowhere near usable as a feature -- it needs a year, and the
    block refuses until it has one. But "this name is being written about three
    times as much as usual" is a fact about today that needs no history at all,
    and it is free, because the collector is already storing the items.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    today = now.date()
    window = now - dt.timedelta(days=NEWS_WINDOW_DAYS)

    out = []
    for symbol in symbols:
        items = news_mod._read(symbol)
        if len(items) < MIN_ITEMS_FOR_SPIKE:
            continue

        seen = []
        for item in items:
            try:
                seen.append(dt.datetime.fromisoformat(
                    item["captured_utc"].replace("Z", "+00:00")))
            except (KeyError, TypeError, ValueError):
                continue

        recent = [when for when in seen if when >= window]
        if len(recent) < MIN_ITEMS_FOR_SPIKE:
            continue

        days = max((now - min(recent)).days, 1)
        usual = len(recent) / days
        todays = sum(1 for when in recent if when.date() == today)
        if usual > 0 and todays >= MIN_ITEMS_FOR_SPIKE and todays / usual >= NEWS_SPIKE:
            out.append({"symbol": symbol, "today": todays,
                        "usual": round(usual, 2),
                        "times": round(todays / usual, 1)})

    return sorted(out, key=lambda item: -item["times"])[:5]


def _book_lines() -> list:
    """What the books are doing, said plainly."""
    lines = []
    funded = holding_mod.funded()

    if funded is None:
        lines.append({"kind": "stopped", "text":
                      "Nothing is funded: trading is stopped and the books are "
                      "recording only."})
    for book_id, book in holding_mod.books().items():
        current = holding_mod.state(book_id)
        held = current["holding"]
        who = "the funded book" if book_id == funded else book_id

        if current["plan"] and not held:
            names = ", ".join(b["symbol"] for b in current["plan"]["buy"])
            lines.append({"kind": "plan", "text":
                          f"{who} wants to buy {names} at the next open."})
        elif held:
            watch = current.get("watch") or {}
            names = ", ".join(b["symbol"] for b in held["bought"])
            if watch.get("status") == "broken":
                lines.append({"kind": "broken", "text":
                              f"{who} still holds {names}, but the case for it "
                              f"has gone: {watch.get('note')}. It holds anyway "
                              f"-- acting on this was measured and lost."})
            elif watch.get("status") == "drifting":
                lines.append({"kind": "drifting", "text":
                              f"{who} holds {names} and its case is weakening: "
                              f"{watch.get('note')}."})
    return lines


def _contest_lines() -> list:
    from . import promote as promote_mod

    verdict = promote_mod.decide()
    lines = []
    for challenger in verdict.get("challengers", []):
        pair = challenger.get("paired") or {}
        if pair.get("sessions", 0) >= MIN_CONTEST_SESSIONS:
            lines.append({"kind": "contest", "text":
                          f"{challenger['book_id']} is {pair['ahead']:+.1%} "
                          f"against the funded book over {pair['sessions']} "
                          f"shared sessions, paired t {pair['t']:+.2f}."})
    if verdict.get("promote"):
        lines.append({"kind": "promotion", "text": verdict["why"]})
    return lines


def write(symbols=None, *, now: dt.datetime | None = None) -> dict | None:
    """Write today's note, if there is anything to say. Once a day.

    Silence is a legitimate outcome and the common one: a book that is holding
    quietly, an archive with no spikes and a contest with nothing to report is
    a day with no news, and saying so every morning would train anybody reading
    to stop.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    today = now.date().isoformat()
    if any(note.get("date") == today for note in read(limit=5)):
        return None

    lines = _book_lines() + _contest_lines()

    spikes = news_spikes(symbols or [], now=now)
    for spike in spikes:
        lines.append({"kind": "news", "text":
                      f"{spike['symbol']} is in {spike['today']} articles today "
                      f"against {spike['usual']} a day lately -- "
                      f"{spike['times']}x its own normal."})

    interesting = [line for line in lines
                   if line["kind"] in ("broken", "news", "promotion", "stopped",
                                       "contest", "drifting")]
    if not interesting and not any(line["kind"] == "plan" for line in lines):
        return None

    note = {"date": today, "written_utc": now.isoformat(timespec="seconds"),
            "lines": lines}
    logger.info("Briefing for %s: %d line(s)", today, len(lines))
    return _append(note)
