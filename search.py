"""Search configurations on a validation period, then open the seal once.

    python search.py run --macro on,off --horizon 1,5    # score a grid
    python search.py board                               # the leaders, and the honest bar
    python search.py commit 12 --why "best validation sharpe, simplest spec"
    python search.py open                                # the sealed period, counted

Compute stopped being the scarce thing a while ago. A model trains in seconds,
so trying three hundred configurations is an afternoon. What that afternoon
spends is the test set: every configuration scored against it is a question
asked of the same rows, and after a few hundred questions the best answer is
noise wearing a good hat.

So the panel is cut in three -- train, validation, test -- and the search is
structurally unable to reach the third. `run` scores on validation only, with
the sealed period truncated out before a model is fitted. Every trial goes into
an append-only ledger, and `open` refuses to score anything until one
configuration has been committed in writing, with a reason, while the answer is
still unknown. Deciding before looking is the whole difference between a test
and a search.

The trial count sets the bar. After four hundred looks, one of them clearing
two standard errors by luck is close to certain, so `board` and `open` both
print what an edge has to beat given how many times you have looked.
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "env", ".env"))

from trader import dataset, labels, pipeline, prices, universe   # noqa: E402
from trader import search as search_mod                          # noqa: E402

logger = logging.getLogger("search")

# Past this, the correction eats most of what a search could find. Not a limit,
# a warning: the ledger will record trial 5,000 as faithfully as trial 5.
MANY_TRIALS = 500


def _numbers(raw: str, cast=int) -> list:
    return [cast(piece.strip()) for piece in raw.split(",") if piece.strip()]


def _switch(raw: str) -> list:
    """`on,off` becomes [True, False]; `on` becomes [True]."""
    truth = {"on": True, "true": True, "yes": True, "1": True,
             "off": False, "false": False, "no": False, "0": False}
    out = []
    for piece in raw.split(","):
        piece = piece.strip().lower()
        if piece not in truth:
            raise argparse.ArgumentTypeError(f"{piece!r} is not on or off")
        out.append(truth[piece])
    return out


def _targets(raw: str) -> list:
    chosen = [piece.strip() for piece in raw.split(",") if piece.strip()]
    for target in chosen:
        if target not in labels.TARGETS:
            raise argparse.ArgumentTypeError(
                f"{target!r} is not one of {sorted(labels.TARGETS)}")
    return chosen


def _load(watchlist) -> dict:
    symbols = universe.resolve(watchlist) if watchlist else pipeline.default_watchlist()
    logger.info("Loading %d symbols", len(symbols))
    frames = prices.load_many(symbols, period="10y")
    if not frames:
        raise SystemExit("No price history could be fetched for any symbol.")
    return frames


def _bar_for(entry: dict, trials: int) -> dict:
    """The corrected bar at this trial count, sized by this trial's evidence."""
    scores = entry.get("validation") or {}
    # Older ledger lines predate effective_rows. Raw rows overstate the
    # evidence, so the bar they give is too low -- say so rather than guess.
    rows = scores.get("effective_rows") or scores.get("rows") or 1
    bar = search_mod.corrected_threshold(
        trials, baseline=float(scores.get("baseline") or 0.5),
        effective_rows=int(rows))
    bar["from_raw_rows"] = not scores.get("effective_rows")
    return bar


# --- printing ---------------------------------------------------------------

def _blocks(spec: dict) -> str:
    return "".join(letter for letter, key in
                   (("M", "use_macro"), ("C", "use_cross"),
                    ("E", "use_events"), ("N", "use_news"))
                   if spec.get(key)) or "-"


def _print_board(rows: list) -> None:
    if not rows:
        print("  no trials yet")
        return
    print(f"\n  {'trial':>5}  {'edge':>8}  {'sharpe':>7}  {'t':>6}  configuration")
    for entry in rows:
        scores = entry.get("validation") or {}
        spec = entry.get("spec") or {}
        print(f"  {entry['trial']:>5}  {scores.get('edge') or 0.0:>+8.4f}  "
              f"{scores.get('executable_sharpe') or 0.0:>+7.2f}  "
              f"{scores.get('executable_tstat') or 0.0:>+6.2f}  "
              f"{spec.get('target', '?')} h{spec.get('horizon', '?')} "
              f"band {spec.get('neutral_band', 0)} [{_blocks(spec)}]"
              + (f"  -- {entry['note']}" if entry.get("note") else ""))
    print("\n  sharpe and t are open -> close, the window an order can reach. "
          "M/C/E/N = macro, cross, events, news.")


# --- the subcommands --------------------------------------------------------

def do_run(args) -> int:
    options = {}
    for name, key in (("macro", "use_macro"), ("cross", "use_cross"),
                      ("events", "use_events")):
        if getattr(args, name) is not None:
            options[key] = getattr(args, name)
    if args.horizon:
        options["horizon"] = _numbers(args.horizon)
    if args.target:
        options["target"] = args.target
    if args.neutral_band:
        options["neutral_band"] = _numbers(args.neutral_band, float)

    if not options:
        print("Nothing to vary. Give at least one of --macro, --cross, "
              "--events, --horizon, --target or --neutral-band.")
        return 2

    combinations = search_mod.grid(**options)
    already = search_mod.count_trials()
    print(f"{len(combinations)} configurations, on top of {already} already "
          f"in the ledger.")
    if already + len(combinations) > MANY_TRIALS:
        print(f"  Past {MANY_TRIALS} trials the correction eats most of what a "
              f"search can find. `search.py board` shows the bar.")

    out = search_mod.run_search(_load(args.universe), combinations,
                                note=args.note or "")

    print(f"\nRan {out['ran']} of {len(combinations)}; "
          f"{out['total_trials']} trials in the ledger.")
    _print_board(out["best"])
    return 0 if out["ran"] else 1


def do_board(args) -> int:
    trials = search_mod.count_trials()
    best = search_mod.leaderboard(args.limit)

    print(f"{trials} trials in the ledger, ranked by validation Sharpe.")
    _print_board(best)

    if best:
        leader = best[0]
        bar = _bar_for(leader, trials)
        edge = (leader.get("validation") or {}).get("edge") or 0.0
        print(f"\n  After {trials} looks an edge has to clear "
              f"{bar['corrected']:+.4f}, where {bar['naive']:+.4f} would have "
              f"done on the first look ({bar['inflation']:.2f}x).")
        if bar["from_raw_rows"]:
            print("  (sized from raw rows -- this trial predates effective "
                  "rows, so the real bar is higher)")
        if edge < bar["corrected"]:
            print(f"  The leader's {edge:+.4f} does not clear it. So far this "
                  f"is the search finding noise, not the model finding an edge.")
        else:
            print(f"  The leader's {edge:+.4f} clears it on validation. That "
                  f"earns a commit, not a conclusion.")

    state = search_mod.seal_state()
    committed = state.get("committed")
    print()
    if committed:
        print(f"  Committed after {committed['after_trials']} trials, because: "
              f"{committed['why']}")
    else:
        print('  Nothing committed. `search.py commit <trial> --why "..."` '
              'before the test set will open.')
    if state.get("opened"):
        print(f"  The test set has been opened {len(state['opened'])} time(s).")
    return 0


def do_commit(args) -> int:
    entry = next((t for t in search_mod.read_trials()
                  if t.get("trial") == args.trial), None)
    if entry is None:
        print(f"No trial {args.trial}. `search.py board` lists them.")
        return 1

    state = search_mod.seal_state()
    if state.get("opened"):
        print(f"Note: the test set has already been opened "
              f"{len(state['opened'])} time(s). Committing again is allowed, "
              f"and the next opening will be reported as a search, not a test.")

    spec = dataset.Spec.from_dict(entry["spec"])
    committed = search_mod.commit(spec, why=args.why)
    print(f"Committed trial {args.trial} after {committed['after_trials']} "
          f"trials.")
    print(f"  {spec.target}, horizon {spec.horizon}, band {spec.neutral_band}, "
          f"blocks [{_blocks(entry['spec'])}]")
    print(f"  because: {args.why}")
    print("\n`search.py open` scores it on the sealed period.")
    return 0


def do_open(args) -> int:
    state = search_mod.seal_state()
    committed = state.get("committed")
    if not committed:
        print("Nothing has been committed, so there is nothing to test. "
              "Choose on validation first, then `search.py commit`.")
        return 1

    if state.get("opened") and not args.again:
        print(f"The test set has already been opened "
              f"{len(state['opened'])} time(s). Opening it again turns the "
              f"test into a slower search. Pass --again if that is what you "
              f"mean.")
        return 1

    spec = dataset.Spec.from_dict(committed["spec"])
    prepared = dataset.prepare(_load(args.universe), spec)
    cut = search_mod.three_way_cut(prepared, test_fraction=spec.test_fraction)

    out = search_mod.open_test_set(prepared, cut)
    scores = out["scores"]

    print(f"\nSealed period from {cut.test_start.date()}: "
          f"{scores['rows']:,} rows over {scores['days']} sessions "
          f"({scores['effective_rows']:,} effective).")
    print(f"  accuracy {scores['accuracy']:.4f} against "
          f"{scores['baseline_accuracy']:.4f}   edge {scores['edge']:+.4f}")
    print(f"  open -> close  sharpe {scores['executable_sharpe']:+.2f}  "
          f"t {scores['executable_tstat']:+.2f}  "
          f"annualised {scores['executable_annualised']:+.1%}")
    print(f"\n{out['reading']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--universe", default=None,
                        help=f"one of {sorted(universe.TIERS)} "
                             f"(default: $TRADER_UNIVERSE or wide)")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="score a grid on validation")
    run.add_argument("--macro", type=_switch, help="on, off, or on,off")
    run.add_argument("--cross", type=_switch, help="on, off, or on,off")
    run.add_argument("--events", type=_switch, help="on, off, or on,off")
    run.add_argument("--horizon", help="comma-separated, e.g. 1,5,20")
    run.add_argument("--target", type=_targets,
                     help=f"comma-separated from {sorted(labels.TARGETS)}")
    run.add_argument("--neutral-band", dest="neutral_band",
                     help="comma-separated, e.g. 0,0.002")
    run.add_argument("--note", help="what you were trying, kept in the ledger")
    run.set_defaults(func=do_run)

    board = commands.add_parser("board", help="the leaders and the honest bar")
    board.add_argument("--limit", type=int, default=10)
    board.set_defaults(func=do_board)

    commit = commands.add_parser("commit", help="declare what goes to the test set")
    commit.add_argument("trial", type=int)
    commit.add_argument("--why", required=True,
                        help="the reason, written before the answer is known")
    commit.set_defaults(func=do_commit)

    opener = commands.add_parser("open", help="score the committed spec on the sealed period")
    opener.add_argument("--again", action="store_true",
                        help="open a test set that has already been opened")
    opener.set_defaults(func=do_open)

    args = parser.parse_args()
    # Logging goes to stderr unbuffered; without this, piped output prints the
    # summary after the trials it summarises.
    sys.stdout.reconfigure(line_buffering=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
