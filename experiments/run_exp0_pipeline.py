"""
Experiment 0 Pipeline Runner
============================

PURPOSE
-------
Single entry point to run Experiment 0 variants and optionally generate graphs.

RUN MODES
---------
    - baseline      : Baseline vs Random
    - move-ordering : Move Ordering vs Random
    - transposition : Transposition vs Random
    - baseline-vs-transposition : Baseline vs Transposition
    - all           : run baseline, move-ordering, and transposition

EXAMPLES
--------
    python -m experiments.run_exp0_pipeline --run baseline
    python -m experiments.run_exp0_pipeline --run move-ordering --depth 5 --games 100
    python -m experiments.run_exp0_pipeline --run transposition --depth 4 --games 50
    python -m experiments.run_exp0_pipeline --run baseline-vs-transposition --depth 4 --games 50
    python -m experiments.run_exp0_pipeline --run all --depth 4 --games 50
    python -m experiments.run_exp0_pipeline --run all --no-plot
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from typing import Any

from experiments.plot_performance import DEFAULT_IMAGES_DIR, ExperimentDataset, build_graph_images


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RUN_CHOICES = [
    "baseline",
    "move-ordering",
    "transposition",
    "baseline-vs-transposition",
    "all",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_experiment_functions(run_mode: str):
    if run_mode == "baseline":
        from experiments.exp0_baseline_vs_random import run_experiment, save_results

        return run_experiment, save_results, "Baseline", "baseline_vs_random"

    if run_mode == "move-ordering":
        from experiments.exp0_move_ordering_vs_random import run_experiment, save_results

        return run_experiment, save_results, "Move Ordering", "move_ordering_vs_random"

    if run_mode == "transposition":
        from experiments.exp0_transposition_vs_random import run_experiment, save_results

        return run_experiment, save_results, "Transposition", "transposition_vs_random"

    raise ValueError(f"Unsupported run mode: {run_mode}")


def _build_dataset(label: str, records: list[Any], summary: Any) -> ExperimentDataset:
    return ExperimentDataset(
        label=label,
        games=[asdict(r) for r in records],
        summary=asdict(summary),
    )


def _save_combined_stats(
    output_root: str,
    depth: int,
    games: int,
    baseline_summary: dict[str, Any],
    transposition_summary: dict[str, Any],
) -> tuple[str, str]:
    os.makedirs(output_root, exist_ok=True)

    baseline_nodes = float(baseline_summary.get("avg_nodes_per_move_mean", 0.0))
    transposition_nodes = float(transposition_summary.get("avg_nodes_per_move_mean", 0.0))
    baseline_time = float(baseline_summary.get("avg_time_per_move_ms_mean", 0.0))
    transposition_time = float(transposition_summary.get("avg_time_per_move_ms_mean", 0.0))
    baseline_win = float(baseline_summary.get("win_rate_pct", 0.0))
    transposition_win = float(transposition_summary.get("win_rate_pct", 0.0))

    node_reduction_pct = 0.0
    if baseline_nodes > 0:
        node_reduction_pct = (1.0 - transposition_nodes / baseline_nodes) * 100.0

    time_reduction_pct = 0.0
    if baseline_time > 0:
        time_reduction_pct = (1.0 - transposition_time / baseline_time) * 100.0

    combined = {
        "run_mode": "both",
        "baseline_depth": depth,
        "num_games": games,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "baseline_vs_random": baseline_summary,
        "transposition_vs_random": transposition_summary,
        "comparison": {
            "win_rate_delta_pct": transposition_win - baseline_win,
            "nodes_per_move_delta": transposition_nodes - baseline_nodes,
            "time_per_move_ms_delta": transposition_time - baseline_time,
            "transposition_nodes_reduction_pct_vs_baseline": node_reduction_pct,
            "transposition_time_reduction_pct_vs_baseline": time_reduction_pct,
        },
    }

    run_folder = f"d{depth}_n{games}"
    run_dir = os.path.join(output_root, run_folder)
    os.makedirs(run_dir, exist_ok=True)

    run_path = os.path.join(run_dir, "exp0_combined_stats.json")
    latest_path = os.path.join(output_root, "exp0_combined_stats.json")

    with open(run_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    return run_path, latest_path


def _save_three_way_stats(
    output_root: str,
    depth: int,
    games: int,
    baseline_summary: dict[str, Any],
    ordering_summary: dict[str, Any],
    transposition_summary: dict[str, Any],
) -> tuple[str, str]:
    os.makedirs(output_root, exist_ok=True)

    combined = {
        "run_mode": "all",
        "baseline_depth": depth,
        "num_games": games,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "baseline_vs_random": baseline_summary,
        "move_ordering_vs_random": ordering_summary,
        "transposition_vs_random": transposition_summary,
    }

    run_folder = f"d{depth}_n{games}"
    run_dir = os.path.join(output_root, run_folder)
    os.makedirs(run_dir, exist_ok=True)

    run_path = os.path.join(run_dir, "exp0_three_way_comparison.json")
    latest_path = os.path.join(output_root, "exp0_three_way_comparison.json")

    with open(run_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    return run_path, latest_path


def _run_single_mode(
    mode: str,
    num_games: int,
    depth: int,
    time_limit: float,
    verbose: bool,
    no_save: bool,
    plot_label: str,
) -> tuple[ExperimentDataset, str, dict[str, Any], Any]:
    run_experiment, save_results, default_label, run_prefix = _resolve_experiment_functions(mode)

    records, summary = run_experiment(
        num_games=num_games,
        depth=depth,
        time_limit=time_limit,
        verbose=verbose,
    )

    if not no_save:
        save_results(records, summary)

    label = plot_label or default_label
    dataset = _build_dataset(label, records, summary)
    summary_dict: dict[str, Any] = asdict(summary)

    return dataset, run_prefix, summary_dict, summary


def _print_saved_graphs(title: str, paths: list[str]) -> None:
    print(f"\n{title}")
    for path in paths:
        print(f"  {path}")


def _summary_depth(summary: Any) -> int:
    return int(getattr(summary, "baseline_depth", getattr(summary, "transposition_depth", 0)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exp0 experiments and generate graph images"
    )

    parser.add_argument(
        "--run",
        type=str,
        choices=RUN_CHOICES,
        default="baseline",
        help="Select a single agent experiment, a baseline-vs-transposition compare, or all",
    )

    parser.add_argument("--games", type=int, default=50, help="Total games to play")
    parser.add_argument("--depth", type=int, default=4, help="Search depth")
    parser.add_argument("--time-limit", type=float, default=5.0, help="Time limit per move (s)")
    parser.add_argument("--verbose", action="store_true", help="Print board state each move")

    parser.add_argument("--no-save", action="store_true", help="Skip writing experiment JSON files")
    parser.add_argument("--no-plot", action="store_true", help="Skip graph generation")
    parser.add_argument(
        "--plot-output-dir",
        type=str,
        default=DEFAULT_IMAGES_DIR,
        help="Root output directory (matchup and run folders are created underneath)",
    )
    parser.add_argument(
        "--plot-label",
        type=str,
        default="",
        help="Optional dataset label used in graph legends (default depends on matchup)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.run in {"baseline", "move-ordering", "transposition"}:
        dataset, run_prefix, _, summary_obj = _run_single_mode(
            mode=args.run,
            num_games=args.games,
            depth=args.depth,
            time_limit=args.time_limit,
            verbose=args.verbose,
            no_save=args.no_save,
            plot_label=args.plot_label,
        )

        if args.no_plot:
            return

        run_folder = f"d{_summary_depth(summary_obj)}_n{summary_obj.total_games}"
        output_root = os.path.join(args.plot_output_dir, run_prefix, run_folder, "images")
        output_paths = build_graph_images([dataset], output_root, run_folder)
        _print_saved_graphs("Saved performance graphs:", output_paths)
        return

    if args.run == "baseline-vs-transposition":
        baseline_dataset, _, baseline_summary, _ = _run_single_mode(
            mode="baseline",
            num_games=args.games,
            depth=args.depth,
            time_limit=args.time_limit,
            verbose=args.verbose,
            no_save=args.no_save,
            plot_label=args.plot_label,
        )
        transposition_dataset, _, transposition_summary, _ = _run_single_mode(
            mode="transposition",
            num_games=args.games,
            depth=args.depth,
            time_limit=args.time_limit,
            verbose=args.verbose,
            no_save=args.no_save,
            plot_label=args.plot_label,
        )

        if not args.no_save:
            comparison_root = os.path.join("results", "exp0", "baseline_vs_transposition")
            run_path, latest_path = _save_combined_stats(
                output_root=comparison_root,
                depth=args.depth,
                games=args.games,
                baseline_summary=baseline_summary,
                transposition_summary=transposition_summary,
            )
            print("\nSaved combined stats:")
            print(f"  {run_path}")
            print(f"  {latest_path}")

        if args.no_plot:
            return

        run_folder = f"d{args.depth}_n{args.games}"
        comparison_images_root = os.path.join(
            args.plot_output_dir,
            "baseline_vs_transposition",
            run_folder,
            "images",
        )
        output_paths = build_graph_images(
            [baseline_dataset, transposition_dataset],
            comparison_images_root,
            run_folder,
        )
        _print_saved_graphs("Saved comparison graphs:", output_paths)
        return

    baseline_dataset, _, baseline_summary, _ = _run_single_mode(
        mode="baseline",
        num_games=args.games,
        depth=args.depth,
        time_limit=args.time_limit,
        verbose=args.verbose,
        no_save=args.no_save,
        plot_label=args.plot_label,
    )
    ordering_dataset, _, ordering_summary, _ = _run_single_mode(
        mode="move-ordering",
        num_games=args.games,
        depth=args.depth,
        time_limit=args.time_limit,
        verbose=args.verbose,
        no_save=args.no_save,
        plot_label=args.plot_label,
    )
    transposition_dataset, _, transposition_summary, _ = _run_single_mode(
        mode="transposition",
        num_games=args.games,
        depth=args.depth,
        time_limit=args.time_limit,
        verbose=args.verbose,
        no_save=args.no_save,
        plot_label=args.plot_label,
    )

    if not args.no_save:
        run_path, latest_path = _save_three_way_stats(
            output_root=os.path.join("results", "exp0", "three_way_comparison"),
            depth=args.depth,
            games=args.games,
            baseline_summary=baseline_summary,
            ordering_summary=ordering_summary,
            transposition_summary=transposition_summary,
        )
        print("\nSaved combined stats:")
        print(f"  {run_path}")
        print(f"  {latest_path}")

    if args.no_plot:
        return

    run_folder = f"d{args.depth}_n{args.games}"
    comparison_images_root = os.path.join(
        args.plot_output_dir,
        "three_way_comparison",
        run_folder,
        "images",
    )
    output_paths = build_graph_images(
        [baseline_dataset, ordering_dataset, transposition_dataset],
        comparison_images_root,
        run_folder,
    )
    _print_saved_graphs("Saved comparison graphs:", output_paths)


if __name__ == "__main__":
    main()
