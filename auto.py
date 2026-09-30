"""Keep training on a schedule, without a browser open.

    python auto.py                          # check every 30 minutes until Ctrl+C
    python auto.py --interval 120           # every two hours
    python auto.py --once                   # one cycle, for Task Scheduler or cron
    python auto.py --stop                   # stop a loop running anywhere else
    python auto.py --status                 # what it is doing

The same loop the Auto-train button starts, in the foreground. Ctrl+C asks it to
stop and waits for the cycle in flight to finish.

It trains when a session has closed since the last time it trained, and not
otherwise -- see src/trader/auto.py for why re-running the same configuration on
unchanged rows is not free. The rest of the cycle is the part that genuinely
wants a schedule and runs every time: collect news, advance the books whose
session has now happened, step the challenger contest, and write the briefing.
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "env", ".env"))

from trader import auto, universe                           # noqa: E402

logger = logging.getLogger("auto")


def describe(state: dict) -> str:
    if not state.get("running"):
        because = state.get("stopped_because")
        return f"Auto-training is off{f' -- {because}' if because else ''}."

    last = state.get("last") or {}
    did = (f"failed: {last['error']}" if last.get("error")
           else f"trained {last['trained']}" if last.get("trained")
           else last.get("skipped") or "starting")
    return (f"Auto-training is on (every {state.get('interval_minutes')} min), "
            f"{state.get('cycles', 0)} cycle(s) so far. Last: {did}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interval", type=float,
                        default=auto.DEFAULT_INTERVAL_MINUTES,
                        help="minutes between checks (default: %(default)s)")
    parser.add_argument("--universe", default=None,
                        help=f"one of {sorted(universe.TIERS)}")
    parser.add_argument("--once", action="store_true",
                        help="run one cycle and exit")
    parser.add_argument("--force", action="store_true",
                        help="train even if no new session has closed")
    parser.add_argument("--stop", action="store_true",
                        help="ask a loop running anywhere to stop")
    parser.add_argument("--status", action="store_true",
                        help="print what it is doing and exit")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")

    if args.status:
        print(describe(auto.state()))
        print(json.dumps(auto.state().get("last") or {}, indent=2))
        return 0

    if args.stop:
        print(describe(auto.stop()))
        return 0

    watchlist = universe.resolve(args.universe) if args.universe else None

    if args.once:
        done = auto.cycle(watchlist=watchlist,
                          trained_for=auto.state().get("trained_for"),
                          force=args.force)
        print(json.dumps(done, indent=2))
        return 0

    auto.start(interval_minutes=args.interval, watchlist=watchlist)
    print(describe(auto.state()))
    print("Ctrl+C to stop; the cycle in flight finishes first.")

    try:
        while auto.running():
            auto._stop.wait(1.0)
    except KeyboardInterrupt:
        print("\nStopping after this cycle...")
        auto.stop(wait=600)

    print(describe(auto.state()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
