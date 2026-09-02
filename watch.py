"""Pick up finished models without anybody watching.

    python watch.py            # every 60 seconds until stopped
    python watch.py --once     # one pass, for cron or Task Scheduler

The UI does the same thing while a page is open. This exists for when it is
not: training takes minutes to hours, and the point of the hand-off is that you
do not have to sit and wait for it.

It is a poll rather than a callback because the coordinator cannot reach back
into a laptop, and opening a port to let it would be a worse trade than asking
every minute. `collect_all` only touches runs that have not resolved, so running
this and the UI at the same time is harmless.
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


def one_pass() -> int:
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
        elif run.status == "failed":
            finished += 1
            logger.warning("%s failed: %s", run.run_id, run.error)
        else:
            logger.info("%s still %s", run.run_id, run.status)

    if not runs:
        logger.debug("nothing outstanding")
    return finished


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true",
                        help="one pass and exit, for a scheduler")
    parser.add_argument("--interval", type=int, default=60,
                        help="seconds between passes (default 60)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.once:
        one_pass()
        return 0

    logger.info("Watching for finished models every %ds. Ctrl-C to stop.",
                args.interval)
    try:
        while True:
            try:
                one_pass()
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
