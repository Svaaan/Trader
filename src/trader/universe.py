"""Which symbols the panel is built from, and why width is the point.

The project started on ten large caps, which is a reasonable place to start and
a bad place to stay. Width is not about having more rows -- it changes what
questions can be asked at all:

**A macro feature needs a cross-section to mean anything.** On ten symbols, a
day when the curve steepens is one observation: every name gets the same value
and the model can only learn "steepening days are up days", which is a claim
about the index measured about four hundred times. On four hundred symbols the
same day is four hundred observations of *which kinds of company* react to
steepening and how, which is a real and learnable question. The macro block goes
from being the smallest part of the dataset to being one of the most useful, and
nothing about the block itself changes.

**A cross-sectional rank needs peers.** "Top half of its peers" among ten names
is a coarse instrument and half of those names are in different industries in
different currencies. Among hundreds it is a percentile that means something.

**And the noise floor falls.** More independent names means the design effect
correction in evaluate.py bites less hard, so the same true edge clears the
trust gate with less evidence.

The cost is the first fetch. Ten symbols take seconds; four hundred take a few
minutes and are then cached, and prices.py now refuses a cached history that is
shorter than the period asked for, so the cache cannot quietly rot the way it
did before.

The lists below are liquid large caps with long histories, which is the point --
a universe assembled from whatever is interesting today is a universe assembled
from survivors. These are not index constituents as of any particular date, and
that is a real limitation: a name that left an index or was acquired is not
here, so the panel is mildly survivor-biased. Fixing that properly needs a
point-in-time constituent history, which is a paid dataset. It is written down
here rather than left for somebody to discover in the results.
"""

from __future__ import annotations

# The original ten. Fast to fetch, useful for iterating on code, and too narrow
# to draw conclusions from -- keep it for the first, not for the answer.
CORE = [
    "AAPL", "MSFT", "NVDA", "JPM", "XOM",
    "ASML.AS", "SAP.DE", "NESN.SW", "MC.PA", "VOLV-B.ST",
]

US = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "BRK-B",
    "JPM", "V", "MA", "WMT", "LLY", "UNH", "XOM", "ORCL", "COST", "HD", "PG",
    "NFLX", "BAC", "ABBV", "KO", "CVX", "CRM", "AMD", "PEP", "TMO", "ADBE",
    "MRK", "CSCO", "ACN", "MCD", "ABT", "WFC", "PM", "DHR", "IBM", "GE",
    "TXN", "QCOM", "NOW", "CAT", "VZ", "INTU", "AXP", "DIS", "AMGN", "PFE",
    "MS", "RTX", "SPGI", "GS", "UNP", "ISRG", "BKNG", "T", "LOW", "BLK",
    "SYK", "TJX", "VRTX", "PGR", "C", "ADP", "MMC", "SCHW", "ELV", "CB",
    "MDT", "ADI", "GILD", "LMT", "AMT", "CI", "SO", "PLD", "DE", "BSX",
    "MO", "ZTS", "DUK", "CME", "ICE", "EQIX", "ITW", "SHW", "APD", "NOC",
    "MMM", "GD", "FDX", "EMR", "PSX", "SLB", "EOG", "MPC", "VLO", "OXY",
    "KMI", "WMB", "NEM", "NSC", "CSX", "AON", "TRV", "ALL", "MET", "PRU",
    "AFL", "KMB", "GIS", "SYY", "HSY", "STZ", "K", "CLX", "CAG", "MKC",
]

EUROPE = [
    # France
    "MC.PA", "OR.PA", "TTE.PA", "SAN.PA", "AIR.PA", "SU.PA", "BNP.PA", "AI.PA",
    "EL.PA", "RMS.PA", "DG.PA", "SGO.PA", "KER.PA", "ORA.PA", "VIV.PA",
    "ENGI.PA", "ACA.PA", "CAP.PA", "LR.PA", "PUB.PA",
    # Germany
    "SAP.DE", "SIE.DE", "ALV.DE", "DTE.DE", "BAS.DE", "BAYN.DE", "BMW.DE",
    "MBG.DE", "VOW3.DE", "ADS.DE", "MUV2.DE", "IFX.DE", "DBK.DE", "RWE.DE",
    "HEN3.DE", "MRK.DE", "EOAN.DE", "FRE.DE", "CON.DE", "HEI.DE",
    # Netherlands
    "ASML.AS", "INGA.AS", "AD.AS", "PHIA.AS", "HEIA.AS", "WKL.AS", "DSFIR.AS",
    "AKZA.AS", "KPN.AS", "RAND.AS",
    # Switzerland
    "NESN.SW", "NOVN.SW", "ROG.SW", "ZURN.SW", "UBSG.SW", "ABBN.SW", "CFR.SW",
    "LONN.SW", "SIKA.SW", "GIVN.SW", "SCMN.SW", "GEBN.SW",
    # Nordics
    "VOLV-B.ST", "ERIC-B.ST", "ATCO-A.ST", "INVE-B.ST", "SEB-A.ST", "HM-B.ST",
    "ASSA-B.ST", "SAND.ST", "NDA-SE.ST", "SKF-B.ST", "TEL2-B.ST", "ALFA.ST",
    "NOVO-B.CO", "MAERSK-B.CO", "DSV.CO", "CARL-B.CO", "TRYG.CO",
    "EQNR.OL", "DNB.OL", "NHY.OL", "TEL.OL", "ORK.OL",
    # Spain, Italy, Belgium, Portugal
    "ITX.MC", "SAN.MC", "BBVA.MC", "IBE.MC", "REP.MC", "AENA.MC", "FER.MC",
    "ISP.MI", "ENI.MI", "ENEL.MI", "G.MI", "UCG.MI", "STLAM.MI", "RACE.MI",
    "ABI.BR", "KBC.BR", "UCB.BR",
    # United Kingdom
    "SHEL.L", "AZN.L", "HSBA.L", "ULVR.L", "BP.L", "GSK.L", "RIO.L", "DGE.L",
    "BATS.L", "LSEG.L", "REL.L", "NG.L", "BARC.L", "LLOY.L", "PRU.L", "TSCO.L",
    "VOD.L", "IMB.L", "AAL.L", "GLEN.L",
]

# The default for a real run. Wide enough that a macro feature has a
# cross-section and a rank has peers; still all large, liquid names whose daily
# close is a real price rather than the last trade somebody happened to make.
WIDE = US + EUROPE

TIERS = {"core": CORE, "us": US, "europe": EUROPE, "wide": WIDE}


def resolve(name_or_symbols) -> list:
    """A tier name, or a list of symbols, into a de-duplicated symbol list.

    Order is preserved so that a run's watchlist reads the way it was written,
    and duplicates are dropped because a symbol appearing twice in a panel would
    be counted twice in every cross-sectional statistic.
    """
    if isinstance(name_or_symbols, str):
        key = name_or_symbols.strip().lower()
        if key not in TIERS:
            raise ValueError(
                f"unknown universe {name_or_symbols!r}; try {sorted(TIERS)}")
        symbols = TIERS[key]
    else:
        symbols = list(name_or_symbols)

    seen, out = set(), []
    for symbol in symbols:
        if symbol not in seen:
            seen.add(symbol)
            out.append(symbol)
    return out
