"""Pick up finished models, and keep the news store growing.

    python watch.py                 # every 60 seconds until stopped
    python watch.py --once          # one pass, for cron or Task Scheduler
    python watch.py --news          # also append to the point-in-time store
    python watch.py --news-only     # just the store, no coordinator calls

The UI does the collecting half while a page is open. This exists for when it is
not: training takes minutes to hours, and the point of the hand-off is that you
do not have to sit and wait for it.

The news half is different in kind and worth understanding before switching it
on. A news archive fetched later has been re-ranked by what turned out to
matter, and its publication timestamps are frequently ingestion times -- so a
model trained on one has been shown which stories mattered, which is the thing
it was supposed to work out. The store therefore records when *this machine*
provably saw each item, and it can only be built forwards. Start it now and the
block becomes usable in about a year; do not start it and the option never
exists. That is the real cost of doing this honestly, and it is cheap to pay
early.

It is a poll rather than a callback because the coordinator cannot reach back
into a laptop, and opening a port to let it would be a worse trade than asking
every minute. `collect_all` only touches runs that have not resolved, so running
this and the UI at the same time is harmless, and the news pass is idempotent by
item id.
"""

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "env", ".env"))

from trader import pipeline                        # noqa: E402

logger = logging.getLogger("watch")


def collect_models() -> int:
    """Collect anything finished. Returns how many runs reached a conclusion."""
    runs = pipeline.collect_all()
    finished = 0

    for run in runs:
        if run.status == "done":
            finished += 1
            evaluation = run.evaluation or {}
            logger.info("%s finished: accuracy %.4f against baseline %.4f (edge %+.4f)",
                        run.run_id, evaluation.get("accuracy", float("nan")),
                        evaluation.get("baseline_accuracy", float("nan")),
                        evaluation.get("edge", float("nan")))
            logger.info("  %s", run.verdict)

            # The controls are the point of comparison, so print them here too
            # rather than leaving them for whoever opens the page.
            controls = run.controls or {}
            for name in ("majority", "logistic", "local_mlp"):
                result = controls.get(name)
                if result:
                    logger.info("  %-10s accuracy %.4f (edge %+.4f)",
                                name, result.get("accuracy", float("nan")),
                                result.get("edge", float("nan")))
            floor = controls.get("noise_floor") or {}
            if floor.get("spread") is not None:
                logger.info("  seed spread %.4f -- an edge below this is a seed",
                            floor["spread"])

        elif run.status == "failed":
            finished += 1
            logger.warning("%s failed: %s", run.run_id, run.error)
        else:
            logger.info("%s still %s", run.run_id, run.status)

    if not runs:
        logger.debug("nothing outstanding")
    return finished


def settle_paper() -> None:
    """Fill any paper entry whose session has now happened.

    On the same timer as everything else. The ledger records an intent in the
    evening and the price it is filled at does not exist until the next open,
    so something has to come back later and close the loop.
    """
    try:
        result = pipeline.settle_paper()
    except Exception:                               # noqa: BLE001
        logger.exception("paper settlement failed, continuing")
        return

    if result.get("settled"):
        logger.info("paper: settled %d day(s), %d still pending",
                    result["settled"], result.get("pending", 0))


def collect_news() -> None:
    """One append-only pass over the store."""
    try:
        result = pipeline.collect_news()
    except Exception:                               # noqa: BLE001
        logger.exception("news pass failed, continuing")
        return

    state = result["readiness"]
    if result["added"]:
        logger.info("news: +%d items across %d symbols (%d total, %d/%d days)",
                    result["added"], result["symbols"], state["items"],
                    state["history_days"], state["needed_days"])
    else:
        logger.debug("news: nothing new (%d stored)", state["items"])


def one_pass(*, models: bool = True, news: bool = False,
             paper: bool = True) -> int:
    finished = collect_models() if models else 0
    if paper:
        settle_paper()
    if news:
        collect_news()
    return finished


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true",
                        help="one pass and exit, for a scheduler")
    parser.add_argument("--interval", type=int, default=60,
                        help="seconds between passes (default 60)")
    parser.add_argument("--news", action="store_true",
                        help="also append to the point-in-time news store")
    parser.add_argument("--news-only", action="store_true",
                        help="only the news store; no coordinator calls")
    parser.add_argument("--no-paper", action="store_true",
                        help="skip settling the paper ledger")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    models = not args.news_only
    news = args.news or args.news_only

    if args.once:
        one_pass(models=models, news=news, paper=not args.no_paper)
        return 0

    what = " and ".join(filter(None, ["finished models" if models else "",
                                      "news" if news else ""]))
    logger.info("Watching for %s every %ds. Ctrl-C to stop.", what, args.interval)
    try:
        while True:
            try:
                one_pass(models=models, news=news, paper=not args.no_paper)
            except Exception:                       # noqa: BLE001
                # A network blip should not end the watch; the next pass will
                # pick up whatever this one missed.
                logger.exception("pass failed, continuing")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        logger.info("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
