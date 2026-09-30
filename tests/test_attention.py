"""Reading the archive, and the rules that keep the reading honest.

The news block cannot be a feature for a year. That makes this layer tempting
and dangerous in the same breath: it is the only thing that can say something
about two thousand stored items today, and it is one import away from becoming
an input to the model it is not allowed to touch.

So the tests come in two kinds. The first is the seam -- nothing that builds a
feature may import this. The second is about the claims themselves, and most of
them exist because the first version of this module made exactly the mistake
they now forbid.
"""

import datetime as dt
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trader import attention, briefing, news                  # noqa: E402


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(news, "STORE_DIR", str(tmp_path / "news"))
    monkeypatch.setattr(briefing, "BRIEFING_DIR", str(tmp_path / "briefing"))
    return tmp_path


def item(number, *, published, captured, title="something happened",
         provider="Wire A", summary=""):
    return {"id": f"item-{number}", "title": title, "summary": summary,
            "provider": provider, "url": "http://example.invalid",
            "published_utc": published.isoformat(),
            "captured_utc": captured.isoformat(timespec="seconds")}


def stock_up(symbol, items):
    path = news._store_path(symbol)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for entry in items:
            handle.write(json.dumps(entry) + "\n")


def spread(symbol, *, days, per_day=1, now=None, title="routine coverage",
           provider="Wire A"):
    """A symbol with a real publication history, so it has a normal."""
    now = now or dt.datetime.now(dt.timezone.utc)
    # One capture moment for all of them, which is what a collector pass does.
    captured = now
    entries = []
    for day in range(days):
        when = now - dt.timedelta(days=day)
        for n in range(per_day):
            entries.append(item(f"{symbol}-{day}-{n}", published=when,
                                captured=captured, title=title,
                                provider=provider))
    stock_up(symbol, entries)
    return entries


# --- the seam ------------------------------------------------------------------

def test_nothing_that_builds_features_can_read_the_archive():
    """It reports on the stores; it must never become an input to them."""
    import ast
    import inspect

    from trader import cross, dataset, evaluate, features, labels, macro, news as n

    for module in (dataset, features, labels, macro, cross, evaluate, n):
        tree = ast.parse(inspect.getsource(module))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.add(node.module or "")
                names.update(alias.name for alias in node.names)
        assert not {n for n in names if "attention" in n}, module.__name__


# --- publication time, not capture time ---------------------------------------

def test_a_first_visit_is_not_a_news_spike(store):
    """The bug this whole distinction exists for.

    The collector stores whatever the provider is showing -- ten items, all
    captured in the same second. Counted by capture, a symbol the collector
    has just met has ten articles "today" against a usual of nearly zero, and
    reads as a tenfold spike forever. Five names were being reported that way.
    """
    now = dt.datetime.now(dt.timezone.utc)
    # Ten items published over two months, all captured in one pass just now.
    stock_up("FRESH", [
        item(n, published=now - dt.timedelta(days=n * 6), captured=now)
        for n in range(10)
    ])

    assert briefing.news_spikes(["FRESH"], now=now) == []


def test_a_real_burst_is_still_reported(store):
    """The guard must not simply silence the feature it protects."""
    now = dt.datetime.now(dt.timezone.utc)
    quiet = [item(f"q{n}", published=now - dt.timedelta(days=n), captured=now)
             for n in range(2, 30)]
    burst = [item(f"b{n}", published=now, captured=now) for n in range(9)]
    stock_up("LOUD", quiet + burst)

    spikes = briefing.news_spikes(["LOUD"], now=now)
    assert [s["symbol"] for s in spikes] == ["LOUD"]
    assert spikes[0]["times"] >= briefing.NEWS_SPIKE


def test_the_archive_is_read_by_publication_not_capture(store):
    """`news.build` counts captures and must keep doing so; reading does not."""
    now = dt.datetime.now(dt.timezone.utc)
    stock_up("SPLIT", [
        item(n, published=now - dt.timedelta(days=n * 3), captured=now)
        for n in range(8)
    ])

    archive = attention._archive(["SPLIT"])
    stamps = {when.date() for when, _ in archive["SPLIT"]}
    assert len(stamps) == 8, "all eight collapsed onto the capture day"


# --- claims it refuses to make ------------------------------------------------

def test_no_multiple_is_claimed_without_a_baseline(store):
    """A multiple against no normal is a number with a times sign in it."""
    now = dt.datetime.now(dt.timezone.utc)
    stock_up("THIN", [item(n, published=now, captured=now) for n in range(9)])

    assert attention.loudest(["THIN"], now=now) == []
    assert attention.without_baseline(["THIN"], now=now) == 1

    rows = {row["market"]: row for row in attention.markets(["THIN"], now=now)}
    assert rows["United States"]["articles"] == 9
    assert rows["United States"]["times"] is None, "claimed a normal it has none of"


def test_a_name_at_its_usual_volume_is_not_loud(store):
    """It was listing a name at 0.7x under a heading that said it stood out."""
    now = dt.datetime.now(dt.timezone.utc)
    spread("STEADY", days=40, per_day=1, now=now)

    assert [row["symbol"] for row in attention.loudest(["STEADY"], now=now)] == []


def test_a_name_well_above_its_own_normal_is_loud(store):
    now = dt.datetime.now(dt.timezone.utc)
    entries = spread("RISING", days=40, per_day=1, now=now)
    entries += [item(f"x{n}", published=now, captured=now) for n in range(12)]
    stock_up("RISING", entries)

    loud = attention.loudest(["RISING"], now=now)
    assert [row["symbol"] for row in loud] == ["RISING"]
    assert loud[0]["times"] > attention.QUIET_ENOUGH


# --- suggestions --------------------------------------------------------------

def test_the_same_company_on_another_exchange_is_not_a_suggestion(store):
    """The first version suggested four names, all already tracked elsewhere.

    GSK on the NYSE against the tracked GSK.L is one company with two lines,
    and telling somebody who follows the London listing to look at the ADR is
    not a suggestion, it is a rounding error in the ticker convention.
    """
    now = dt.datetime.now(dt.timezone.utc)
    stock_up("GSK.L", [
        item(n, published=now - dt.timedelta(days=n), captured=now,
             title=f"GSK plc (NYSE:GSK) reports something, day {n}",
             provider=f"Wire {n % 3}")
        for n in range(5)
    ])

    assert attention.suggestions(["GSK.L"], now=now) == []


def test_a_name_seen_in_one_companys_coverage_is_not_corroborated(store):
    """A second listing shows up in exactly one company's stories."""
    now = dt.datetime.now(dt.timezone.utc)
    stock_up("PRU.L", [
        item(n, published=now - dt.timedelta(days=n), captured=now,
             title=f"Prudential (NYSE:PUK) does a thing, day {n}",
             provider=f"Wire {n % 3}")
        for n in range(5)
    ])

    found = attention.suggestions(["PRU.L"], now=now)
    assert [row["symbol"] for row in found] == ["PUK"]
    assert found[0]["corroborated"] is False
    assert found[0]["contexts"] == 1


def test_a_name_in_several_companies_coverage_is_a_suggestion(store):
    """Turning up in unrelated stories is what separates a company from a line."""
    now = dt.datetime.now(dt.timezone.utc)
    for source in ("AAPL", "MSFT", "NVDA"):
        stock_up(source, [
            item(f"{source}{n}", published=now - dt.timedelta(days=n),
                 captured=now,
                 title=f"{source} and the chip maker (NASDAQ:ARM) day {n}",
                 provider=f"Wire {n % 3}")
            for n in range(3)
        ])

    found = attention.suggestions(["AAPL", "MSFT", "NVDA"], now=now)
    assert [row["symbol"] for row in found] == ["ARM"]
    assert found[0]["corroborated"] is True
    assert found[0]["contexts"] == 3
    assert found[0]["headlines"], "a suggestion with no evidence is a guess"


# --- markets -------------------------------------------------------------------

def test_a_market_with_tracked_names_is_not_a_gap(store):
    now = dt.datetime.now(dt.timezone.utc)
    stock_up("SHEL.L", [
        item(n, published=now - dt.timedelta(days=n), captured=now,
             title=f"BP (LSE:BP.) and others, day {n}", provider=f"Wire {n % 3}")
        for n in range(4)
    ])

    assert attention.unwatched_markets(["SHEL.L"], now=now) == []


def test_a_market_named_repeatedly_with_nothing_tracked_is_a_gap(store):
    """Two different companies, because one story repeated is not a market."""
    now = dt.datetime.now(dt.timezone.utc)
    stock_up("AAL.L", [
        item(0, published=now - dt.timedelta(days=1), captured=now,
             title="Teck Resources (TSX:TECK.B) merger terms", provider="Wire A"),
        item(1, published=now - dt.timedelta(days=2), captured=now,
             title="Manulife (TSX:MFC) signs reinsurance deal", provider="Wire B"),
        item(2, published=now - dt.timedelta(days=3), captured=now,
             title="Manulife (TSX:MFC) looks reasonable", provider="Wire A"),
    ])

    gaps = attention.unwatched_markets(["AAL.L"], now=now)
    assert [gap["market"] for gap in gaps] == ["Toronto"]
    assert gaps[0]["distinct_names"] == 2
    assert gaps[0]["headlines"]


def test_one_company_repeated_is_not_a_market(store):
    now = dt.datetime.now(dt.timezone.utc)
    stock_up("AAL.L", [
        item(n, published=now - dt.timedelta(days=n), captured=now,
             title=f"Teck Resources (TSX:TECK.B) again, day {n}",
             provider=f"Wire {n % 3}")
        for n in range(5)
    ])

    assert attention.unwatched_markets(["AAL.L"], now=now) == []


def test_the_market_of_a_symbol_comes_from_its_suffix():
    assert attention.market_of("AAPL") == "United States"
    assert attention.market_of("SHEL.L") == "London"
    assert attention.market_of("VOLV-B.ST") == "Stockholm"
    assert attention.market_of("NOVO-B.CO") == "Copenhagen"


# --- the reading ---------------------------------------------------------------

def test_an_empty_archive_says_so_rather_than_inventing(store):
    out = attention.opinion(["AAPL", "MSFT"])
    assert out["reading"]
    assert any(line["kind"] == "quiet" for line in out["reading"])
    assert out["suggestions"] == []
    assert out["loudest"] == []


def test_the_reading_always_carries_what_it_was_computed_from(store):
    now = dt.datetime.now(dt.timezone.utc)
    spread("AAPL", days=40, per_day=1, now=now)

    out = attention.opinion(["AAPL"], now=now)
    assert str(out["readiness"]["items"]) in out["caveat"].replace(",", "")
    assert "reading, not evidence" in out["caveat"]
