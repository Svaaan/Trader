"""What it noticed today, and the discipline that keeps that worth reading.

Everything else in the project is built for a verdict months or years away.
This is the output that is useful on the day -- so the failure mode is not
being wrong, it is being noise. A note every morning saying nothing trains
anybody reading it to stop, and a note that claims more than was measured is
worse than no note at all.
"""

import datetime as dt
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from conftest import synthetic_panel                          # noqa: E402
from trader import briefing, holding, news                    # noqa: E402


@pytest.fixture
def stores(tmp_path, monkeypatch):
    monkeypatch.setattr(briefing, "BRIEFING_DIR", str(tmp_path / "briefing"))
    monkeypatch.setattr(holding, "STORE_DIR", str(tmp_path / "books"))
    monkeypatch.setattr(news, "STORE_DIR", str(tmp_path / "news"))
    return tmp_path


class Run:
    run_id = "run-1"
    trust = {"trusted": False, "reason": "test"}

    def __init__(self, as_of, picks):
        self.signals = [{"symbol": s, "as_of": as_of, "probability_up": p,
                         "confidence": abs(p - 0.5) * 2} for s, p in picks]


def article(when, number):
    return {"id": f"item-{number}", "title": f"something {number}",
            "summary": "", "published_utc": when.isoformat(),
            "provider": "test", "url": "http://example.invalid",
            "captured_utc": when.isoformat(timespec="seconds")}


def stock_up(symbol, items):
    path = news._store_path(symbol)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item) + "\n")


# --- silence is the common case ------------------------------------------------

def test_a_quiet_day_writes_nothing(stores):
    """A note every morning saying nothing teaches you to stop reading them."""
    assert briefing.write([]) is None
    assert briefing.read() == []


def test_it_says_what_the_book_wants_to_buy(stores, tmp_path):
    panel = synthetic_panel()
    symbols = sorted(panel)
    as_of = list(panel[symbols[0]].index)[-10].date().isoformat()
    holding.plan_next(Run(as_of, [(symbols[0], 0.9)]))

    note = briefing.write([])
    assert note is not None
    assert any(symbols[0] in line["text"] and line["kind"] == "plan"
               for line in note["lines"])


def test_it_writes_once_a_day(stores, tmp_path):
    panel = synthetic_panel()
    symbols = sorted(panel)
    as_of = list(panel[symbols[0]].index)[-10].date().isoformat()
    holding.plan_next(Run(as_of, [(symbols[0], 0.9)]))

    assert briefing.write([]) is not None
    assert briefing.write([]) is None, "it wrote twice in a day"

    tomorrow = dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=1)
    assert briefing.write([], now=tomorrow) is not None


# --- what it is allowed to say -------------------------------------------------

def test_a_broken_case_is_reported_and_the_holding_is_not(stores):
    """The one thing worth saying about a position that has lost its reason is
    that it lost it -- and that nothing was done, because doing something was
    measured and lost."""
    holding._append({"fill": {
        "at": "2026-09-01T00:00:00+00:00", "as_of": "2026-09-01",
        "run_id": "r", "book": holding.BOOK, "trusted": False,
        "session": "2026-09-02", "cash_before": 500.0, "cash_after": 0.0,
        "fees": 2.5, "bought": [{"symbol": "AAA", "session": "2026-09-02",
                                 "price": 10.0, "value": 500.0, "fee": 2.5,
                                 "shares": 49.75, "probability_up": 0.9,
                                 "pattern": [{"feature": "rsi_14",
                                              "effect": 0.03, "z": -1.0}]}]}})
    holding._append({"watch": {"at": "2026-09-10T00:00:00+00:00",
                               "symbol": "AAA", "session": "2026-09-10",
                               "status": "broken", "note": "2 of 3 reasons have gone",
                               "probability_now": 0.44, "probability_at_entry": 0.9,
                               "agreement": 0.33, "kept": [], "lost": []}})

    note = briefing.write([])
    broken = [line for line in note["lines"] if line["kind"] == "broken"][0]
    assert "2 of 3 reasons have gone" in broken["text"]
    assert "holds anyway" in broken["text"]


def test_a_name_suddenly_in_the_news_is_worth_a_line(stores):
    """The archive is a year away from being a feature. "Three times its own
    normal" is a fact about today and needs no history at all."""
    now = dt.datetime.now(dt.timezone.utc)
    quiet = [article(now - dt.timedelta(days=day), day) for day in range(2, 30, 3)]
    loud = quiet + [article(now, 100 + n) for n in range(8)]
    stock_up("LOUD", loud)
    stock_up("QUIET", quiet)

    spikes = briefing.news_spikes(["LOUD", "QUIET"], now=now)
    assert [s["symbol"] for s in spikes] == ["LOUD"]
    assert spikes[0]["times"] >= briefing.NEWS_SPIKE


def test_a_name_that_is_always_in_the_news_is_not_a_spike(stores):
    now = dt.datetime.now(dt.timezone.utc)
    always = []
    for day in range(30):
        when = now - dt.timedelta(days=day)
        always += [article(when, day * 10 + n) for n in range(6)]
    stock_up("NOISY", always)

    assert briefing.news_spikes(["NOISY"], now=now) == []


def test_a_stopped_book_says_so_first(stores):
    holding.stand_down(why="past the -20% stop")
    note = briefing.write([])

    assert note["lines"][0]["kind"] == "stopped"
    assert "trading is stopped" in note["lines"][0]["text"].lower()


def test_the_notes_are_appended_and_dated(stores):
    holding.stand_down(why="testing")
    briefing.write([])
    path = briefing._path()
    before = open(path, encoding="utf-8").read()

    tomorrow = dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=1)
    briefing.write([], now=tomorrow)
    after = open(path, encoding="utf-8").read()

    assert after.startswith(before)
    assert len(briefing.read()) == 2
    assert briefing.read()[0]["date"] == tomorrow.date().isoformat()


def test_nothing_that_builds_features_can_read_the_briefing():
    """It reports on the record; it must never become an input to it."""
    import ast
    import inspect

    from trader import cross, dataset, evaluate, features, labels, macro

    for module in (dataset, features, labels, macro, cross, evaluate):
        tree = ast.parse(inspect.getsource(module))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.add(node.module or "")
                names.update(alias.name for alias in node.names)
        assert not {n for n in names if "briefing" in n}, module.__name__
