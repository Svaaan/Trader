"""What a trade actually costs at a real broker, minimum fee and all.

The rest of this project charges five basis points a side, which is a decent
proxy for spread and slippage and says nothing about commission. A retail
Nordic broker does not charge in basis points: it charges a percentage **or a
minimum, whichever is larger**, and the minimum is what decides everything when
the positions are small.

The arithmetic that matters here: the paper account is 500 and the book holds
238 names, so a position is about 2.10. Avanza Mini on US stock is 0.25% with a
minimum of 1 USD a side -- so the round trip on a 2.10 position is 2 USD of
commission plus currency, which is about **95% of the position**. The model
would need to be right about the direction of a stock by a factor of two,
every day, to break even. That is not a tuning problem, it is an arithmetic
one, and it is worth seeing before any more effort goes into the signal.

Every number here was read off the brokers' own price pages on 2026-09-28 and
carries its source. They change; `checked` is when, and the note on each one
says what it covers. None of this is advice about which broker to use.
"""

from __future__ import annotations

import dataclasses
import math

# Roughly what one unit of each currency is worth in the account's currency,
# which this project keeps in dollars. Approximate and dated deliberately: a
# 10% error here moves a minimum fee by a few cents and changes no conclusion
# below, where the fees are 30 to 95% of a position. Pass `rates` to override.
DEFAULT_RATES = {
    "USD": 1.0,
    "SEK": 0.105,
    "EUR": 1.17,
    "DKK": 0.157,
    "NOK": 0.099,
    "CAD": 0.72,
}


@dataclasses.dataclass(frozen=True)
class Schedule:
    """One broker's price for one market, as charged per side of a trade."""

    name: str
    rate: float                 # fraction of the trade value
    minimum: float              # per side, in `currency`, whichever is larger
    currency: str
    fx_per_side: float          # markup for converting into the trade currency
    covers: str
    source: str
    checked: str = "2026-09-28"

    def minimum_in_account(self, rates: dict | None = None) -> float:
        rates = {**DEFAULT_RATES, **(rates or {})}
        return self.minimum * rates.get(self.currency, 1.0)


# Avanza, https://www.avanza.se/konton-lan-prislista/prislista/courtageklasser.html
# and .../handel-utland.html. The currency markup is Avanza's own figure: the
# automatic exchange spread is 0.5% round trip, "alltså 0,25% åt vardera håll".
AVANZA_START_SE = Schedule(
    name="Avanza Start (Stockholm)", rate=0.0, minimum=0.0, currency="SEK",
    fx_per_side=0.0, covers="Swedish shares on the Stockholm main list only",
    source="avanza.se/konton-lan-prislista/prislista/courtageklasser.html")

AVANZA_MINI_SE = Schedule(
    name="Avanza Mini (Stockholm)", rate=0.0025, minimum=1.0, currency="SEK",
    fx_per_side=0.0, covers="Swedish shares",
    source="avanza.se/konton-lan-prislista/prislista/courtageklasser.html")

AVANZA_MINI_US = Schedule(
    name="Avanza Mini (US)", rate=0.0025, minimum=1.0, currency="USD",
    fx_per_side=0.0025, covers="US shares, the Start and Mini classes alike",
    source="avanza.se/konton-lan-prislista/prislista/handel-utland.html")

AVANZA_SMALL_US = Schedule(
    name="Avanza Small (US)", rate=0.0015, minimum=6.0, currency="USD",
    fx_per_side=0.0025, covers="US shares",
    source="avanza.se/konton-lan-prislista/prislista/handel-utland.html")

AVANZA_MINI_DE = Schedule(
    name="Avanza Mini (Germany)", rate=0.0025, minimum=1.0, currency="EUR",
    fx_per_side=0.0025, covers="German shares, ETFs and certificates",
    source="avanza.se/konton-lan-prislista/prislista/handel-utland.html")

# Nordnet, https://www.nordnet.se/kundservice/prislista. New customers trade the
# Nordic exchanges free until 2027-06-30 and then fall into Mini, which is why
# the Nordic minimum below is the one that applies afterwards.
NORDNET_MINI_NORDIC = Schedule(
    name="Nordnet Mini (Nordic)", rate=0.0025, minimum=1.0, currency="SEK",
    fx_per_side=0.0, covers="main Nordic exchanges, orders up to 15,600 SEK",
    source="nordnet.se/kundservice/prislista")

NORDNET_MINI_FOREIGN = Schedule(
    name="Nordnet Mini (outside the Nordics)", rate=0.0025, minimum=9.0,
    currency="SEK", fx_per_side=0.0025,
    covers="US and other non-Nordic shares, orders up to 15,600 SEK",
    source="nordnet.se/kundservice/prislista")

NORDNET_LITEN_FOREIGN = Schedule(
    name="Nordnet Liten (outside the Nordics)", rate=0.0015, minimum=49.0,
    currency="SEK", fx_per_side=0.0025, covers="orders of 15,600-46,000 SEK",
    source="nordnet.se/kundservice/prislista")

# What the project charged before this module existed: spread and slippage, no
# commission and no minimum. Kept so the two can be compared.
SPREAD_ONLY = Schedule(
    name="spread only (what the backtest charges)", rate=0.0005, minimum=0.0,
    currency="USD", fx_per_side=0.0, covers="no commission, no minimum",
    source="evaluate.DEFAULT_COST")

SCHEDULES = {
    "spread_only": SPREAD_ONLY,
    "avanza_start_se": AVANZA_START_SE,
    "avanza_mini_se": AVANZA_MINI_SE,
    "avanza_mini_us": AVANZA_MINI_US,
    "avanza_small_us": AVANZA_SMALL_US,
    "avanza_mini_de": AVANZA_MINI_DE,
    "nordnet_mini_nordic": NORDNET_MINI_NORDIC,
    "nordnet_mini_foreign": NORDNET_MINI_FOREIGN,
    "nordnet_liten_foreign": NORDNET_LITEN_FOREIGN,
}

# What a book of this project's shape pays, unless told otherwise: a US-listed
# position at the cheapest class either broker offers for it.
DEFAULT_SCHEDULE = "avanza_mini_us"


def resolve(schedule) -> Schedule:
    """A Schedule from a Schedule, a key, or None for the default."""
    if schedule is None:
        return SCHEDULES[DEFAULT_SCHEDULE]
    if isinstance(schedule, Schedule):
        return schedule
    try:
        return SCHEDULES[str(schedule)]
    except KeyError:
        raise ValueError(
            f"unknown schedule {schedule!r}; try one of {sorted(SCHEDULES)}"
        ) from None


def per_side(value: float, schedule=None, *, rates: dict | None = None) -> float:
    """Commission plus currency markup on one side, in the account's currency.

    `value` is what the position is worth. The percentage and the minimum are
    not added together -- the broker charges whichever is larger -- and the
    currency markup rides on top of both, because it is charged on the money
    being converted rather than on the trade.
    """
    schedule = resolve(schedule)
    value = abs(float(value))
    if value <= 0:
        return 0.0

    commission = max(value * schedule.rate, schedule.minimum_in_account(rates))
    return commission + value * schedule.fx_per_side


def round_trip(value: float, schedule=None, *, rates: dict | None = None) -> float:
    """In and out again: what one complete trade costs."""
    return 2.0 * per_side(value, schedule, rates=rates)


def fraction(value: float, schedule=None, *, rates: dict | None = None) -> float:
    """The round trip as a fraction of the position -- the number that has to
    be earned before anything is earned."""
    value = abs(float(value))
    if value <= 0:
        return 0.0
    return round_trip(value, schedule, rates=rates) / value


def break_even_value(schedule=None, *, rates: dict | None = None) -> float:
    """The position size at which the minimum stops deciding the bill.

    Below this the percentage is irrelevant and every trade costs the same, so
    halving the position doubles the cost as a share of it.
    """
    schedule = resolve(schedule)
    if schedule.rate <= 0:
        return math.inf if schedule.minimum > 0 else 0.0
    return schedule.minimum_in_account(rates) / schedule.rate


def position_value(equity: float, weight: float) -> float:
    return abs(float(equity) * float(weight))


def describe(value: float, schedule=None, *, rates: dict | None = None) -> dict:
    """What one position of this size costs, and what it has to make back."""
    schedule = resolve(schedule)
    side = per_side(value, schedule, rates=rates)
    minimum = schedule.minimum_in_account(rates)
    return {
        "schedule": schedule.name,
        "source": schedule.source,
        "checked": schedule.checked,
        "position_value": round(float(value), 4),
        "commission_per_side": round(side, 4),
        "round_trip": round(2.0 * side, 4),
        "round_trip_fraction": round(fraction(value, schedule, rates=rates), 6),
        "minimum_binds": bool(value * schedule.rate < minimum),
        "break_even_value": round(break_even_value(schedule, rates=rates), 2),
    }
