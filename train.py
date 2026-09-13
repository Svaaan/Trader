"""Train a model, here or on HelloWorldAi or both.

    python train.py                       # both: train here and submit, then compare
    python train.py --backend local       # here only, ~30s, no network
    python train.py --backend helloworld  # submit only, the original path
    python train.py --universe core       # ten symbols, for iterating
    python train.py --target absolute     # the other target
    python train.py --no-controls         # skip the baselines and walk-forward

`--backend both` is the one worth running. The same rows and the same
hyperparameters go to both trainers, so any gap between their scores is a fact
about the round trip rather than about the data. The local half finishes in
about half a minute and is scored immediately, so there is something to read
while the remote job queues; when it lands, `watch.py` or the UI picks it up and
the comparison appears.

Training here needs no GPU and no network. The model is 7,233 parameters and the
full step count is about twenty-six seconds of numpy -- a 3070 is idle at that
size, which is worth knowing before wiring one up.
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "env", ".env"))

from trader import labels, pipeline, trainer, universe      # noqa: E402

logger = logging.getLogger("train")


def report(run) -> None:
    """What happened, in the order it is worth reading."""
    data = run.dataset or {}

    print(f"\nRun {run.run_id}  [{run.backend}]  status: {run.status}")
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
                           ("local_mlp", "local MLP")):
            result = controls.get(key)
            if result:
                print(f"    {label:<12} accuracy {result['accuracy']:.4f}  "
                      f"edge {result['edge']:+.4f}")
        floor = controls.get("noise_floor") or {}
        if floor.get("spread") is not None:
            print(f"    seed spread  {floor['spread']:.4f}  "
                  f"(an edge below this is a seed, not a signal)")

    for label, evaluation in (("local", run.local_evaluation),
                              ("helloworld", run.evaluation)):
        if not evaluation or (label == "helloworld" and run.primary != "helloworld"):
            continue
        print(f"\n  {label} model:")
        print(f"    accuracy {evaluation['accuracy']:.4f} against "
              f"{evaluation['baseline_accuracy']:.4f}  "
              f"edge {evaluation['edge']:+.4f}")
        print(f"    sharpe {evaluation['strategy_sharpe']:+.2f}  "
              f"t {evaluation['strategy_tstat']:+.2f}  "
              f"annualised {evaluation['strategy_annualised']:+.1%}")

    if run.comparison:
        print(f"\n  the two backends: {run.comparison['reading']}")

    if run.trust:
        state = "OPEN" if run.trust.get("trusted") else "SHUT"
        print(f"\n  gate {state}: {run.trust.get('reason')}")

    if run.status == "training":
        print("\n  The remote job is queued. `python watch.py` collects it, "
              "or leave the UI open.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", default=trainer.BOTH,
                        choices=list(trainer.BACKENDS),
                        help="where to train (default: both)")
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
        backend=args.backend,
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
