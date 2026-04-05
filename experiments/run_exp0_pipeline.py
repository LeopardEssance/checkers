"""
Run Experiment 0 and generate performance graphs in one command.

Examples:
    python -m experiments.run_exp0_pipeline
    python -m experiments.run_exp0_pipeline --games 100 --depth 5 --time-limit 3.0
    python -m experiments.run_exp0_pipeline --games 100 --plot-output-dir results/images/doc_ready
    python -m experiments.run_exp0_pipeline --no-plot
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

from experiments.exp0_baseline_vs_random import run_experiment, save_results
from experiments.plot_performance import (
    DEFAULT_IMAGES_DIR,
    ExperimentDataset,
    build_graph_images,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exp0 baseline experiment and generate graph images"
    )

    parser.add_argument("--games", type=int, default=50, help="Total games to play")
    parser.add_argument("--depth", type=int, default=4, help="Baseline search depth")
    parser.add_argument("--time-limit", type=float, default=5.0, help="Time limit per move (s)")
    parser.add_argument("--verbose", action="store_true", help="Print board state each move")

    parser.add_argument("--no-save", action="store_true", help="Skip writing experiment JSON files")
    parser.add_argument("--no-plot", action="store_true", help="Skip graph generation")
    parser.add_argument(
        "--plot-output-dir",
        type=str,
        default=DEFAULT_IMAGES_DIR,
        help="Directory where graph images are saved",
    )
    parser.add_argument(
        "--plot-label",
        type=str,
        default="Baseline",
        help="Dataset label used in graph legends and filenames",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    records, summary = run_experiment(
        num_games=args.games,
        depth=args.depth,
        time_limit=args.time_limit,
        verbose=args.verbose,
    )

    if not args.no_save:
        save_results(records, summary)

    if args.no_plot:
        return

    dataset = ExperimentDataset(
        label=args.plot_label,
        games=[asdict(r) for r in records],
        summary=asdict(summary),
    )
    run_folder = f"exp0_d{summary.baseline_depth}_n{summary.total_games}"
    output_paths = build_graph_images([dataset], args.plot_output_dir, run_folder)

    print("\nSaved performance graphs:")
    for path in output_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
