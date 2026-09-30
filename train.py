"""Train a model and grade it.

    python train.py                     # the wide universe, about a minute
    python train.py --universe core     # ten symbols, for iterating
    python train.py --target absolute   # the other target
    python train.py --no-controls       # skip the baselines and walk-forward

Everything happens here and now: fetch prices, build features, cut the panel
once, train, score on rows the model never saw, run the controls, and print
what the gate makes of it. The run is finished when the command returns.

There used to be a second trainer behind a network. It is gone: the network is
7,233 parameters, a logistic regression matches it, and the round trip was
paying for a difference that was not there.
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "env", ".env"))

from trader import labels, pipeline, universe               # noqa: E402

logger = logging.getLogger("train")


def report(run) -> None:
    """What happened, in the order it is worth reading."""
    data = run.dataset or {}

    print(f"\nRun {run.run_id}  status: {run.status}")
    if run.error:
        print(f"  error: {run.error}")
        return

    print(f"  {len(data.get('symbols') or [])} symbols, "
          f"{len(data.get('feature_names') or [])} features, "
          f"cut {data.get('cut_date')}")
    print(f"  train {(data.get('train') or {}).get('rows', 0):,} rows, "
          f"test {(data.get('test') or {}).get('rows', 0):,} rows")
    if data.get("steps"):
        print(f"  {data['steps']:,} steps = {data.get('epochs')} passes")

    controls = run.controls or {}
    if controls.get("majority"):
        print("\n  what it had to beat:")
        for key, label in (("majority", "majority"), ("logistic", "logistic"),
                           ("local_mlp", "a second MLP")):
            result = controls.get(key)
            if result:
                print(f"    {label:<13} accuracy {result['accuracy']:.4f}  "
                      f"edge {result['edge']:+.4f}")
        floor = controls.get("noise_floor") or {}
        if floor.get("spread") is not None:
            print(f"    seed spread   {floor['spread']:.4f}  "
                  f"(an edge below this is a seed, not a signal)")

    evaluation = run.evaluation or {}
    if evaluation:
        print(f"\n  scored on days it never saw:")
        print(f"    accuracy {evaluation['accuracy']:.4f} against "
              f"{evaluation['baseline_accuracy']:.4f}  "
              f"edge {evaluation['edge']:+.4f}")
        # Two holding windows. The first is what the label describes and what
        # no order can reach; the second is what an order can reach. Printed
        # together because on this panel they disagree completely.
        print(f"    close -> close  sharpe {evaluation['strategy_sharpe']:+.2f}"
              f"  t {evaluation['strategy_tstat']:+.2f}"
              f"  annualised {evaluation['strategy_annualised']:+7.1%}"
              f"   <- not tradeable")
        if evaluation.get("executable_sharpe") is not None:
            print(f"    open  -> close  "
                  f"sharpe {evaluation['executable_sharpe']:+.2f}"
                  f"  t {evaluation['executable_tstat']:+.2f}"
                  f"  annualised {evaluation['executable_annualised']:+7.1%}"
                  f"   <- what the gate reads")
            print(f"    execution gap {evaluation['execution_gap']:+.2f} sharpe,"
                  f" lost to the overnight move")

    if run.trust:
        state = "OPEN" if run.trust.get("trusted") else "SHUT"
        print(f"\n  gate {state}: {run.trust.get('reason')}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--universe", default=None,
                        help=f"one of {sorted(universe.TIERS)} "
                             f"(default: $TRADER_UNIVERSE or wide)")
    parser.add_argument("--target", default=labels.RELATIVE,
                        choices=list(labels.TARGETS),
                        help="what to predict (default: relative)")
    parser.add_argument("--horizon", type=int, default=1,
                        help="sessions ahead to predict (default: 1)")
    parser.add_argument("--steps", type=int, default=None,
                        help="override the derived step count")
    parser.add_argument("--folds", type=int, default=6,
                        help="walk-forward windows (default: 6)")
    parser.add_argument("--no-controls", action="store_true",
                        help="skip the baselines and walk-forward")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")

    run = pipeline.start(
        args.universe,
        target=args.target,
        horizon=args.horizon,
        steps=args.steps,
        folds=args.folds,
        run_controls=not args.no_controls,
    )

    report(run)
    return 0 if run.status != "failed" else 1


if __name__ == "__main__":
    sys.exit(main())
