"""Measure your own follower on the validation fold.

    python matchmaker_eval/test_submission.py submissions/your-name

This is the one command a submitter needs. It downloads the 20 validation
performances if they are not already there, runs your follower over them,
and prints how it did.

It is deliberately limited to the **valid** fold. The eval fold's 146
performances are what the leaderboard is for; using them to develop against is
what the declaration in your metadata.yaml says you did not do, and the easiest
way to keep that true is not to have a convenient way to break it. See
docs/eval-protocol.md.

To measure a built-in method the same way, pass --method instead of a
directory. For anything beyond this -- sharding, other folds, the leaderboard --
use run_submission.py, which this is a thin front door onto.
"""

# Entry-point path setup — see the note in run_submission.py.
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (_REPO_ROOT, _REPO_ROOT / "matchmaker_eval"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import argparse
import numpy as np

from matchmaker_eval.submission import SubmissionError
from matchmaker_eval.utils import pooled_summaries
from run_submission import evaluate_submission

VALID_FOLD = "valid"


def summarise(metrics: dict) -> str:
    """A readable verdict, rather than a path to a JSON file."""
    overall = metrics.get("summary_all") or {}
    tracked = metrics.get("summary_tracked") or {}
    beat_all = overall
    beat_tracked = tracked
    n, total = metrics["n_tracked"], metrics["n_pieces"]

    lines = [
        "",
        f"{metrics['submission']}  ({metrics['input_type']}, {metrics.get('fold', 'validation')} fold)",
        "-" * 65,
        f"  tracking rate      {n}/{total} pieces ({overall.get('tracking_rate', 0.0):.0%})",
    ]
    if beat_tracked.get("mean_b") is not None:
        lines.append(
            f"  beat error (mean)  {beat_tracked['mean_b']:.3f} (tracked) | "
            f"{beat_all.get('mean_b', float('nan')):.3f} (all)"
        )
    if beat_tracked.get("median_b") is not None:
        lines.append(
            f"  beat error (med)   {beat_tracked['median_b']:.3f} (tracked) | "
            f"{beat_all.get('median_b', float('nan')):.3f} (all)"
        )
    if tracked.get("smoothness") is not None:
        lines.append(
            f"  Smoothness       {tracked['smoothness']:.2f} (tracked) | "
            f"{overall.get('smoothness', float('nan')):.2f} (all)"
        )
    if tracked.get("rtf") is not None:
        lines.append(f"  real-time factor   {tracked['rtf']:.4f}")

    by_dataset: dict = {}
    for piece in metrics.get("pieces", []):
        entry = by_dataset.setdefault(piece.get("dataset", "?"), [0, 0])
        entry[0] += int(bool(piece.get("tracked")))
        entry[1] += 1
    if len(by_dataset) > 1:
        lines.append("")
        for dataset, (ok, seen) in sorted(by_dataset.items()):
            lines.append(f"  {dataset:18} {ok}/{seen} tracked")

    lost = [p["piece_id"] for p in metrics.get("pieces", []) if not p.get("tracked")]
    if lost:
        lines.append("")
        lines.append(f"  lost the performance on {len(lost)} piece(s):")
        for piece_id in lost[:5]:
            lines.append(f"    {piece_id}")
        if len(lost) > 5:
            lines.append(f"    ... and {len(lost) - 5} more")

    failures = metrics.get("failures") or []
    if failures:
        lines.append("")
        lines.append(f"  {len(failures)} piece(s) did not run:")
        for failure in failures[:3]:
            lines.append(f"    {failure['piece_id']}: {failure['error'][:70]}")
    return "\n".join(lines)


from matchmaker_eval.methods import available_methods
from tabulate import tabulate


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "submission",
        type=Path,
        nargs="?",
        default=None,
        help="your submission directory, e.g. submissions/your-name",
    )
    parser.add_argument(
        "--method",
        default=None,
        help="measure a built-in method instead of a submission ('all' for all methods)",
    )
    parser.add_argument(
        "--input-type",
        choices=("audio", "midi"),
        default=None,
        help="needed with --method or when running all methods; a submission declares its own",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="stop after the first N pieces, for a quick look",
    )
    parser.add_argument(
        "--plots", action="store_true", help="save a tracking plot per piece"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="where to write the run (default: results/runs/valid/<name>)",
    )
    args = parser.parse_args()

    if args.submission is not None and args.method is not None:
        parser.error("pass either a submission directory or --method, not both")

    if args.submission is None and args.method is None and args.input_type is None:
        parser.error("pass either a submission directory or --method, not both")

    # Determine methods to run
    if args.method and "," in args.method:
        methods_to_run = [m.strip() for m in args.method.split(",") if m.strip()]
        run_multi = True
    elif args.method == "all" or (args.submission is None and args.method is None and args.input_type is not None):
        methods_to_run = available_methods(args.input_type)
        if args.input_type == "midi":
            methods_to_run = [m for m in methods_to_run if m not in ("OPTM", "SL_OLTW")]
        run_multi = True
    elif args.method is not None:
        methods_to_run = [args.method]
        run_multi = False
    elif args.submission is not None:
        methods_to_run = [None]
        run_multi = False
    else:
        parser.error("pass a submission directory, --method <name> (or comma-separated list), or --input-type <audio|midi>")

    if (run_multi or args.method is not None) and not args.input_type and args.submission is None:
        parser.error("pass --input-type <audio|midi> when specifying --method")

    all_metrics = []
    run_dirs = {}
    if run_multi:
        print(f"Running {len(methods_to_run)} methods ({', '.join(methods_to_run)}) for input_type='{args.input_type}' on {VALID_FOLD} fold...")

    for m in methods_to_run:
        try:
            method_run_dir = (
                (args.output / m)
                if (run_multi and args.output is not None and m is not None)
                else args.output
            )
            metrics = evaluate_submission(
                args.submission if args.submission is not None else None,
                fold=VALID_FOLD,
                builtin=m,
                input_type=args.input_type,
                limit=args.limit,
                save_plots=args.plots,
                run_dir=method_run_dir,
            )
            all_metrics.append(metrics)
            run_dirs[metrics["submission"]] = method_run_dir or (
                _REPO_ROOT / "results" / "runs" / VALID_FOLD / metrics["submission"]
            )
            print(summarise(metrics))
        except SubmissionError as e:
            print(f"\n[{m}] Error: {e}", file=sys.stderr)
            if not run_multi:
                return 1

    if run_multi and len(all_metrics) > 1:
        # Compute common tracked subset across all evaluated methods
        tracked_sets = {}
        for met in all_metrics:
            m_name = met.get("method") or met.get("submission")
            tracked_sets[m_name] = set(
                p["index"] for p in met.get("pieces", []) if p.get("tracked")
            )

        common_indices = (
            set.intersection(*tracked_sets.values()) if tracked_sets else set()
        )

        print("\n" + "=" * 70)
        print(
            f"SUMMARY TABLE Across All {len(all_metrics)} Methods (Common Tracked: {len(common_indices)} pieces)"
        )
        print("=" * 70)

        rows = []
        for met in all_metrics:
            m_name = met.get("method") or met.get("submission")
            tr = met.get("summary_all", {}).get("tracking_rate", 0.0)
            tracked_sum = met.get("summary_tracked", {})
            beat_tr = tracked_sum

            # Filter for common tracked pieces
            pieces = met.get("pieces", [])
            common_pieces = [p for p in pieces if p.get("index") in common_indices]
            common_summary = pooled_summaries(
                common_pieces, run_dirs[met["submission"]]
            )["summary_tracked"]

            def _fmt_val(val, fmt_str):
                return "-" if val is None or (isinstance(val, float) and np.isnan(val)) else f"{val:{fmt_str}}"

            def _fmt_pct(val):
                return "-" if val is None or (isinstance(val, float) and np.isnan(val)) else f"{val * 100:.1f}%"

            rows.append(
                {
                    "Method": m_name,
                    "TR (%)": f"{tr * 100:.1f}%",
                    "Tracked": met.get("n_tracked", 0),
                    "MeanAE": _fmt_val(beat_tr.get("mean_b"), ".3f"),
                    "0.5b": _fmt_pct(beat_tr.get("accuracy_b", {}).get("0.5b")),
                    "Smoothness": _fmt_val(tracked_sum.get("smoothness"), ".2f"),
                    "Common Count": len(common_pieces),
                    "Common MeanAE": _fmt_val(common_summary.get("mean_b"), ".3f"),
                    "Common 0.5b": _fmt_pct(common_summary.get("accuracy_b", {}).get("0.5b")),
                    "Common Smoothness": _fmt_val(common_summary.get("smoothness"), ".2f"),
                    "RTF": _fmt_val(tracked_sum.get("rtf"), ".4f"),
                }
            )
        print(tabulate(rows, headers="keys", tablefmt="fancy_grid"))

    print(
        "\nThis is the validation fold. The leaderboard is the eval fold, which is\n"
        "evaluated for you when your pull request is merged."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
