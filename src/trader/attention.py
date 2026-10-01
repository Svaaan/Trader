"""What the archive is loud about, and what it thinks you are missing.

The news store has been collecting for weeks and feeds nothing. `news.build`
refuses to be a feature until it has a year, which is right -- but "not yet a
feature" is not the same as "not yet worth reading", and until now the only
thing done with two thousand stored items was counting them.

So this is the other half: an opinion about the archive, formed the same way
every other opinion here is formed -- **computed from the stores, with the
number behind it**. There is no language model in this file and nothing is
generated. Every line it produces can be traced to items you can open.

Three things it can say that the model cannot:

**Which markets are loud.** A ticker suffix is a market. Grouping coverage by
it turns two thousand scattered items into a handful of statements about where
attention actually is this week, against where it usually is.

**Which names are loud, and what the price did about it.** Coverage and price
are two stores that have never been joined. A name in three times its normal
coverage whose price did nothing is a different thing from one that moved, and
neither is visible from either store alone.

**What it is watching that you are not.** A quarter of stored headlines name a
company explicitly -- "(LSE:AAL)", "(ENXTPA:DG)" -- because the providers write
them that way. Those mentions are extracted, mapped onto the ticker convention
this project uses, and the ones that resolve to something untracked are ranked
by how many separate days and separate providers carried them. A name that nine
outlets mentioned across five days and that nothing in the universe covers is a
suggestion the archive earned, not one anybody invented.

The same extraction answers the larger question. An exchange code that keeps
appearing and maps to a market with **no tracked names at all** is not a missing
stock, it is a missing market.

**What this is not.** It is not a signal, it never becomes a feature, and
nothing here can reach a label or a score -- the same seam drawn around
`briefing.py` and `context.py`, enforced by the same kind of test. A suggestion
is a reading prompt: something to go and look at, with the headlines that
prompted it attached, so it can be dismissed in ten seconds if it is noise.
"""

from __future__ import annotations

import collections
import datetime as dt
import logging
import re

from . import news as news_mod

logger = logging.getLogger(__name__)

# A ticker suffix is a market. Bare tickers are US listings, which is the
# convention yfinance uses and the one universe.py is written in.
MARKETS = {
    "": "United States",
    ".L": "London",
    ".PA": "Paris",
    ".DE": "Frankfurt",
    ".AS": "Amsterdam",
    ".SW": "Switzerland",
    ".ST": "Stockholm",
    ".CO": "Copenhagen",
    ".OL": "Oslo",
    ".MC": "Madrid",
    ".MI": "Milan",
    ".BR": "Brussels",
    ".HE": "Helsinki",
    ".LS": "Lisbon",
    ".VI": "Vienna",
    ".IR": "Dublin",
    ".TO": "Toronto",
    ".AX": "Australia",
    ".T": "Tokyo",
    ".HK": "Hong Kong",
}

# How the providers write an exchange, and the suffix it corresponds to here.
# Several codes reach the same market -- Simply Wall St. says SWX and VTX for
# Switzerland, XTRA and ETR for Xetra -- so the mapping is many-to-one.
EXCHANGES = {
    "NYSE": "", "NASDAQ": "", "NASDAQGS": "", "NYSEAM": "", "AMEX": "",
    "LSE": ".L", "AIM": ".L",
    "ENXTPA": ".PA", "EPA": ".PA",
    "XTRA": ".DE", "ETR": ".DE", "DB": ".DE", "FRA": ".DE",
    "ENXTAM": ".AS", "AMS": ".AS",
    "SWX": ".SW", "VTX": ".SW", "EBR": ".BR", "ENXTBR": ".BR",
    "OM": ".ST", "STO": ".ST", "NGM": ".ST",
    "CPSE": ".CO", "CPH": ".CO",
    "OB": ".OL", "OSL": ".OL",
    "BME": ".MC", "BMAD": ".MC",
    "BIT": ".MI", "MIL": ".MI",
    "HLSE": ".HE", "ELI": ".LS", "WBAG": ".VI", "ISE": ".IR",
    "TSX": ".TO", "TSXV": ".TO",
    "ASX": ".AX", "TSE": ".T", "SEHK": ".HK",
}

# "(LSE:AAL)", "(ENXTPA:DG)", "(TSX:TECK.B)". Parenthesised and upper-case,
# which is how the wires write it and specific enough not to catch prose.
MENTION = re.compile(r"\(([A-Z]{2,8}):([A-Z0-9][A-Z0-9.\-]{0,9})\)")

# Windows. The archive is weeks old, so these are deliberately short: a
# fortnight of "usual" is thin, and the page says so rather than implying a
# year of context it does not have.
RECENT_DAYS = 7
BASELINE_DAYS = 30

# What a suggestion has to clear. Separate days and separate providers, because
# one outlet repeating itself for a week is one source, not five.
MIN_MENTIONS = 3
MIN_DAYS = 2
MIN_PROVIDERS = 2

# Distinct publication days a symbol needs before any "x its own normal" claim
# is made about it. See `_has_baseline`.
MIN_SPREAD_DAYS = 10

# Below this multiple of its own normal a name is not loud, whatever its raw
# count. Names with no baseline yet are ranked by volume instead and say so.
QUIET_ENOUGH = 1.2


def market_of(symbol: str) -> str:
    """The market a ticker belongs to, from its suffix."""
    if "." in symbol:
        suffix = "." + symbol.rsplit(".", 1)[1]
        return MARKETS.get(suffix, suffix.lstrip("."))
    return MARKETS[""]


def _archive(symbols) -> dict:
    """Every stored item per symbol, by publication time, parsed once.

    Publication time and not capture time, which is the opposite of what
    `news.build` does and is right for the opposite reason. `build` feeds the
    model, so it counts captures: a publication timestamp can be revised or
    backfilled, and a feature that reads one can be told about a story after
    the fact. Nothing in this file can reach the model, and for reading, the
    capture time is actively misleading -- this store was filled by three
    collector passes, so counted that way the whole archive happened on three
    days and every symbol the collector meets for the first time looks like a
    ten-fold spike, because ten items is the provider's page size.
    """
    out = {}
    for symbol in symbols:
        items = []
        for item in news_mod._read(symbol):
            when = news_mod.published_at(item)
            if when is not None:
                items.append((when, item))
        if items:
            out[symbol] = sorted(items, key=lambda pair: pair[0])
    return out


def _has_baseline(symbol: str) -> bool:
    """Whether this symbol has enough published history for an 'unusual' claim.

    Same refusal `news.readiness` makes about the feature, at the scale this
    page works on: without a spread of days there is no normal, and a multiple
    against no normal is a number with a multiplication sign in it.
    """
    return news_mod.published_spread(symbol) >= MIN_SPREAD_DAYS


# --- where the attention is ---------------------------------------------------

def markets(symbols, *, now: dt.datetime | None = None) -> list:
    """Coverage per market this week, against its own recent normal.

    Grouped by suffix rather than by anything anybody had to label, so it stays
    correct when the universe changes.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    recent_from = now - dt.timedelta(days=RECENT_DAYS)
    base_from = now - dt.timedelta(days=BASELINE_DAYS)
    archive = _archive(symbols)

    counts: dict = collections.defaultdict(
        lambda: {"recent": 0, "baseline": 0, "symbols": set(), "loud": set()})

    for symbol, items in archive.items():
        market = market_of(symbol)
        entry = counts[market]
        entry["symbols"].add(symbol)
        for when, _ in items:
            if when >= recent_from:
                entry["recent"] += 1
            if when >= base_from:
                entry["baseline"] += 1

    out = []
    for market, entry in counts.items():
        tracked = len(entry["symbols"])
        # Articles per tracked name, so a market with forty names is not
        # automatically louder than one with four.
        per_name = entry["recent"] / tracked if tracked else 0.0

        # A multiple is only offered where the names in this market actually
        # have a history to be unusual against. Where they do not, the count
        # still stands -- it is the comparison that is withheld, not the fact.
        grounded = [s for s in entry["symbols"] if _has_baseline(s)]
        usual = times = None
        if len(grounded) >= max(1, tracked // 2):
            earlier = entry["baseline"] - entry["recent"]
            earlier_days = max(BASELINE_DAYS - RECENT_DAYS, 1)
            usual = (earlier / earlier_days) * RECENT_DAYS if earlier else 0.0
            times = round(entry["recent"] / usual, 2) if usual > 0 else None

        out.append({
            "market": market,
            "tracked": tracked,
            "articles": entry["recent"],
            "per_name": round(per_name, 2),
            "usual": round(usual, 1) if usual is not None else None,
            "times": times,
            "grounded": len(grounded),
        })

    return sorted(out, key=lambda row: -row["per_name"])


def loudest(symbols, frames: dict | None = None, *,
            now: dt.datetime | None = None, limit: int = 8) -> list:
    """Names in unusual coverage, and what their price did about it.

    Two stores that have never been joined. Coverage without the price move is
    a trivia question; the price move without the coverage is a number with no
    story attached. Together they are the one thing on this page that could
    plausibly turn into a question worth asking the model later.

    **Only names with a baseline appear.** A symbol the collector reached for
    the first time has whatever the provider's page size is -- ten, or eleven
    -- and six of those listed together under "the loudest names" is six rows
    of fetch artefact wearing a story's clothes. The count of names left out
    for that reason is reported instead, by `opinion`.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    recent_from = now - dt.timedelta(days=RECENT_DAYS)
    archive = _archive(symbols)

    out = []
    for symbol, items in archive.items():
        recent = [(when, item) for when, item in items if when >= recent_from]
        if len(recent) < MIN_MENTIONS:
            continue

        if not _has_baseline(symbol):
            continue

        earlier = [pair for pair in items if pair[0] < recent_from]
        span = max((recent_from - items[0][0]).days, 1)
        usual = len(earlier) / span * RECENT_DAYS if earlier else 0.0
        times = len(recent) / usual if usual > 0 else None

        move = None
        if frames is not None:
            frame = frames.get(symbol)
            if frame is not None and len(frame) > RECENT_DAYS:
                window = frame["close"].iloc[-(RECENT_DAYS + 1):]
                first = float(window.iloc[0])
                if first:
                    move = round(float(window.iloc[-1]) / first - 1.0, 5)

        # "Loudest" has to mean louder than its own normal: a name was being
        # listed at 0.7x under a heading that said it stood out.
        if times is None or times < QUIET_ENOUGH:
            continue

        out.append({
            "symbol": symbol,
            "market": market_of(symbol),
            "articles": len(recent),
            "usual": round(usual, 1),
            "times": round(times, 1),
            "move": move,
            "headlines": [item.get("title") for _, item in recent[-3:]
                          if item.get("title")],
        })

    # Ranked by how far above its own normal, not by raw volume: a name that
    # always gets forty articles is not news.
    return sorted(out, key=lambda row: -row["times"])[:limit]


def without_baseline(symbols, *, now: dt.datetime | None = None) -> int:
    """Names with coverage this week but too little history to judge it."""
    now = now or dt.datetime.now(dt.timezone.utc)
    recent_from = now - dt.timedelta(days=RECENT_DAYS)
    quiet = 0
    for symbol, items in _archive(symbols).items():
        recent = [pair for pair in items if pair[0] >= recent_from]
        if len(recent) >= MIN_MENTIONS and not _has_baseline(symbol):
            quiet += 1
    return quiet


# --- what it is watching that you are not -------------------------------------

def mentions(symbols, *, now: dt.datetime | None = None) -> dict:
    """Every explicit company mention in the archive, with its evidence.

    Keyed by the symbol the mention maps to under this project's convention,
    so it can be compared directly against the universe.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    archive = _archive(symbols)
    found: dict = {}

    for source, items in archive.items():
        for when, item in items:
            text = f"{item.get('title') or ''} {item.get('summary') or ''}"
            for code, ticker in set(MENTION.findall(text)):
                suffix = EXCHANGES.get(code)
                if suffix is None:
                    continue
                # Share classes: the wires write "TECK.B", the price provider
                # spells it "TECK-B", and the universe already follows the
                # second convention (VOLV-B.ST, NOVO-B.CO). Left as written,
                # the suggestion resolved to TECK.B.TO, which is not a symbol
                # anybody can look up -- a suggestion nobody can act on.
                mapped = f"{ticker.replace('.', '-')}{suffix}"
                entry = found.setdefault(mapped, {
                    "symbol": mapped, "exchange": code,
                    "market": MARKETS.get(suffix, suffix.lstrip(".")),
                    "count": 0, "days": set(), "providers": set(),
                    "alongside": set(), "headlines": [],
                })
                entry["count"] += 1
                entry["days"].add(when.date().isoformat())
                if item.get("provider"):
                    entry["providers"].add(item["provider"])
                entry["alongside"].add(source)
                if item.get("title") and len(entry["headlines"]) < 3:
                    entry["headlines"].append({
                        "title": item["title"], "url": item.get("url"),
                        "provider": item.get("provider"),
                        "seen": when.date().isoformat(),
                    })
    return found


def _root(symbol: str) -> str:
    """The ticker without its market suffix or share-class tail.

    GSK and GSK.L are one company on two exchanges; VOLV-B.ST and VOLV-A.ST
    are two share classes of one company. Neither is a name worth suggesting
    to somebody who already tracks the other.
    """
    stem = symbol.split(".", 1)[0]
    return stem.rsplit("-", 1)[0] if "-" in stem else stem


def suggestions(symbols, *, now: dt.datetime | None = None,
                limit: int = 8) -> list:
    """Names the archive keeps naming that the universe does not cover.

    Two filters do the work, and both were added after reading what the first
    version produced -- which was four suggestions, all of them companies
    already tracked under a different listing:

    **The same company on another exchange is not a new name.** PUK is
    Prudential's US line against the tracked PRU.L; FER and FER.MC are one
    Ferrovial. A shared root ticker catches most of these outright.

    **A name that only ever appears beside one tracked symbol is probably that
    symbol.** Cross-listings that do not share a root -- MRSH beside MMC --
    show up in articles filed under exactly one company, because that is whose
    story they are. A genuinely separate company turns up in more than one
    context, so appearing beside two or more tracked names is what promotes a
    mention from a footnote to a suggestion.

    The bar is otherwise separate days and separate providers rather than raw
    count, because one outlet running a series for a week is one source.
    """
    tracked = set(symbols)
    roots = {_root(symbol) for symbol in tracked}

    out = []
    for symbol, entry in mentions(symbols, now=now).items():
        if symbol in tracked or _root(symbol) in roots:
            continue
        if not (entry["count"] >= MIN_MENTIONS
                and len(entry["days"]) >= MIN_DAYS
                and len(entry["providers"]) >= MIN_PROVIDERS):
            continue

        alongside = sorted(entry["alongside"])
        out.append({
            "symbol": symbol,
            "exchange": entry["exchange"],
            "market": entry["market"],
            "mentions": entry["count"],
            "days": len(entry["days"]),
            "providers": len(entry["providers"]),
            "alongside": alongside[:4],
            "contexts": len(alongside),
            # One context means the archive has only ever seen this name in
            # one company's coverage, which is what a second listing looks
            # like. Kept and marked rather than dropped, because "the name you
            # track also trades here" is worth a line, just not a suggestion.
            "corroborated": len(alongside) >= 2,
            "headlines": entry["headlines"],
            "tracked_in_market": sum(
                1 for s in tracked if market_of(s) == entry["market"]),
        })

    return sorted(out, key=lambda row: (not row["corroborated"],
                                        -row["contexts"], -row["days"],
                                        -row["providers"]))[:limit]


def unwatched_markets(symbols, *, now: dt.datetime | None = None) -> list:
    """Markets the archive keeps naming where nothing at all is tracked.

    The larger version of the same question. A name you do not hold is a gap;
    a market you have no names in is a different kind of gap, and only this
    file is in a position to notice it -- the model cannot miss what it was
    never given a column for.
    """
    tracked_markets = {market_of(symbol) for symbol in symbols}
    rollup: dict = collections.defaultdict(
        lambda: {"mentions": 0, "names": set(), "days": set(),
                 "providers": set(), "headlines": []})

    for entry in mentions(symbols, now=now).values():
        if entry["market"] in tracked_markets:
            continue
        group = rollup[entry["market"]]
        group["mentions"] += entry["count"]
        group["names"].add(entry["symbol"])
        group["days"] |= entry["days"]
        group["providers"] |= entry["providers"]
        for headline in entry["headlines"]:
            if len(group["headlines"]) < 3:
                group["headlines"].append(headline)

    out = [{
        "market": market,
        "mentions": group["mentions"],
        "names": sorted(group["names"])[:6],
        "distinct_names": len(group["names"]),
        "days": len(group["days"]),
        "providers": len(group["providers"]),
        "headlines": group["headlines"],
    } for market, group in rollup.items()
        if group["mentions"] >= MIN_MENTIONS
        and len(group["days"]) >= MIN_DAYS
        # Two different companies, not one story repeated. The claim here is
        # about a market rather than a name, so a second company in it is the
        # corroboration that matters -- which is why this is not the same bar
        # as the one `suggestions` applies to providers.
        and len(group["names"]) >= 2]

    return sorted(out, key=lambda row: (-row["distinct_names"], -row["mentions"]))


# --- the reading ---------------------------------------------------------------

def opinion(symbols, frames: dict | None = None, *,
            now: dt.datetime | None = None) -> dict:
    """Everything above, plus the sentences that follow from it.

    The wording leans towards saying there is nothing here, for the same reason
    `evaluate.verdict` does: this is read by the person who would like it to
    work, so "the archive has nothing to say today" has to be the easy answer
    rather than the one it avoids.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    state = news_mod.readiness(symbols)

    by_market = markets(symbols, now=now)
    loud = loudest(symbols, frames, now=now)
    ideas = suggestions(symbols, now=now)
    unjudged = without_baseline(symbols, now=now)
    new_markets = unwatched_markets(symbols, now=now)

    reading = []

    if state["items"] == 0:
        reading.append({"kind": "quiet", "text":
                        "The archive is empty. Run a collection pass and this "
                        "page starts having something to say."})

    busiest = by_market[0] if by_market else None
    if busiest and busiest["articles"]:
        line = (f"{busiest['market']} is the busiest market in the archive this "
                f"week: {busiest['articles']} articles across "
                f"{busiest['tracked']} tracked names, "
                f"{busiest['per_name']} each.")
        if busiest["times"] and busiest["times"] >= 1.5:
            line += f" That is {busiest['times']}x its own recent normal."
        elif busiest["times"] is None:
            line += (" Too little history stored to say whether that is "
                     "unusual for it.")
        reading.append({"kind": "market", "text": line})

    if loud:
        top = loud[0]
        line = (f"{top['symbol']} is in {top['articles']} articles this week "
                f"against {top['usual']} usually -- {top['times']}x its own "
                f"normal.")
        if top["move"] is not None:
            direction = "rose" if top["move"] > 0 else "fell"
            line += (f" The price {direction} {abs(top['move']) * 100:.1f}% over "
                     f"the same stretch.")
            # The interesting case is the one that did not move, and saying so
            # is the difference between a reading and a headline.
            if abs(top["move"]) < 0.01:
                line += (" Coverage without a price move is the more interesting "
                         "of the two cases and the harder one to act on.")
        reading.append({"kind": "name", "text": line})

    corroborated = [idea for idea in ideas if idea["corroborated"]]
    for idea in corroborated[:3]:
        reading.append({"kind": "suggestion", "text":
                        f"{idea['symbol']} ({idea['market']}) is named in "
                        f"{idea['mentions']} stored items across {idea['days']} "
                        f"days and {idea['providers']} providers, and in the "
                        f"coverage of {idea['contexts']} different tracked "
                        f"names -- and nothing in the universe covers it. Worth "
                        f"a look."})

    single = [idea for idea in ideas if not idea["corroborated"]]
    if single and not corroborated:
        reading.append({"kind": "suggestion-weak", "text":
                        f"No name has come up in enough separate contexts to "
                        f"suggest yet. {len(single)} appeared in only one "
                        f"company's coverage each "
                        f"({', '.join(idea['symbol'] for idea in single[:3])}), "
                        f"which is usually what a second listing of a name "
                        f"already tracked looks like."})

    for gap in new_markets[:2]:
        reading.append({"kind": "market-gap", "text":
                        f"{gap['market']} keeps appearing: {gap['distinct_names']} "
                        f"different companies over {gap['days']} days, and the "
                        f"universe has no names listed there at all."})

    if unjudged:
        reading.append({"kind": "quiet", "text":
                        f"{unjudged} other tracked names had coverage this week "
                        f"with too little stored history to say whether it was "
                        f"unusual. They are left out rather than ranked by raw "
                        f"count, which would only rank them by when the "
                        f"collector first reached them."})

    if not reading:
        reading.append({"kind": "quiet", "text":
                        "Nothing in the archive stands out today. That is the "
                        "common case and it is reported rather than filled."})

    # The caveat travels with the reading rather than sitting in a footnote,
    # because everything above is computed on weeks of history, not years.
    caveat = (f"Computed from {state['items']:,} items captured over "
              f"{state['history_days']} days. The block needs "
              f"{state['needed_days']} before it may become a feature, so "
              f"none of this reaches the model -- it is reading, not evidence.")

    return {
        "at": now.isoformat(timespec="seconds"),
        "readiness": state,
        "markets": by_market,
        "loudest": loud,
        "without_baseline": unjudged,
        "suggestions": ideas,
        "unwatched_markets": new_markets,
        "reading": reading,
        "caveat": caveat,
    }
