"""
Create readable graph images from existing experiment result JSON files.

This script does NOT run experiments. It only reads JSON files already present in
results folders and emits improved charts inside a nested folder under each
existing images directory:

    results/.../images/readable/

Usage:
    python -m experiments.generate_readable_graphs
    python -m experiments.generate_readable_graphs --results-root results
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Any

import matplotlib.pyplot as plt


DEFAULT_RESULTS_ROOT = "results"
READABLE_SUBDIR = "readable"

PALETTE = [
    "#0B6E4F",
    "#33658A",
    "#F26419",
    "#7A306C",
    "#2E4057",
    "#9A031E",
    "#5F0F40",
    "#3A7D44",
]


@dataclass
class PlotTarget:
    name: str
    run_dir: str
    images_dir: str
    games_path: str
    summary_path: str


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def _quantile(sorted_values: list[float], q: float) -> float:
    """
    Compute the q-th quantile (0 <= q <= 1) using linear interpolation.
    Assumes sorted_values is already sorted in ascending order.
    """
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    pos = (len(sorted_values) - 1) * q
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return float(sorted_values[low])

    frac = pos - low
    return float(sorted_values[low] * (1.0 - frac) + sorted_values[high] * frac)


def _binned_stats(values: list[float], max_points: int = 70) -> tuple[list[int], list[float], list[float], list[float], int]:
    """
    Aggregate values into bins to support readable line plots.
    
    Returns: (bin_end_indices, bin_means, bin_25th_percentiles, bin_75th_percentiles, bin_size)
    This allows plotting mean ± IQR (interquartile range) as bands around a line.
    """
    if not values:
        return [], [], [], [], 1

    n = len(values)
    bin_size = max(1, math.ceil(n / max_points))

    xs: list[int] = []
    means: list[float] = []
    p25: list[float] = []
    p75: list[float] = []

    for start in range(0, n, bin_size):
        chunk = values[start : start + bin_size]
        if not chunk:
            continue
        sorted_chunk = sorted(chunk)
        xs.append(min(n, start + len(chunk)))
        means.append(sum(chunk) / len(chunk))
        p25.append(_quantile(sorted_chunk, 0.25))
        p75.append(_quantile(sorted_chunk, 0.75))

    return xs, means, p25, p75, bin_size


def _short_label(text: str, max_len: int = 30) -> str:
    clean = " ".join(str(text).split())
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 1] + "…"


def _discover_targets(results_root: str) -> list[PlotTarget]:
    """
    Recursively find all experiment result folders that contain both
    game JSON files and summary JSON files.
    
    Returns: sorted list of PlotTargets (each represents a run folder with games + summary)
    """
    candidates = {
        "experiment1_head_to_head": ("experiment1_head_to_head.json", "experiment1_summary.json"),
        "experiment2_ablation": ("experiment2_ablation.json", "experiment2_summary.json"),
        "experiment3_scalability": ("experiment3_scalability.json", "experiment3_summary.json"),
    }

    targets: list[PlotTarget] = []

    for experiment_name, (games_file, summary_file) in candidates.items():
        base_dir = os.path.join(results_root, experiment_name)
        if not os.path.isdir(base_dir):
            continue

        for root, dirs, files in os.walk(base_dir):
            if games_file not in files or summary_file not in files:
                continue

            targets.append(
                PlotTarget(
                    name=experiment_name,
                    run_dir=root,
                    images_dir=os.path.join(root, "images"),
                    games_path=os.path.join(root, games_file),
                    summary_path=os.path.join(root, summary_file),
                )
            )

    return sorted(targets, key=lambda target: target.run_dir)


def _save_figure(fig: plt.Figure, output_path: str) -> None:
    """Finalize and save matplotlib figure, then close it to free memory."""
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_stacked_rates(rows: list[dict[str, Any]], label_key: str, title: str, output_path: str) -> None:
    """
    Plot outcome rates (wins/losses/draws) as a stacked bar chart.
    Useful for summarizing match results across multiple conditions.
    """
    labels = [_short_label(row.get(label_key, "Unknown")) for row in rows]
    win_rates = [float(row.get("win_rate_pct", 0.0)) for row in rows]
    loss_rates = [float(row.get("loss_rate_pct", 0.0)) for row in rows]
    draw_rates = [float(row.get("draw_rate_pct", 0.0)) for row in rows]

    x = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 1.6), 5.6))

    ax.bar(x, win_rates, color="#2E8B57", label="Win %")
    ax.bar(x, draw_rates, bottom=win_rates, color="#7A7A7A", label="Draw %")
    ax.bar(
        x,
        loss_rates,
        bottom=[win_rates[i] + draw_rates[i] for i in range(len(rows))],
        color="#B22222",
        label="Loss %",
    )

    ax.set_title(title)
    ax.set_ylabel("Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    _save_figure(fig, output_path)


def _plot_box_distribution(
    games: list[dict[str, Any]],
    group_key: str,
    metric_key: str,
    title: str,
    y_label: str,
    output_path: str,
) -> None:
    buckets: dict[str, list[float]] = defaultdict(list)

    for game in games:
        if group_key not in game or metric_key not in game:
            continue
        try:
            buckets[str(game[group_key])].append(float(game[metric_key]))
        except (TypeError, ValueError):
            continue

    if not buckets:
        return

    ordered = sorted(buckets.items(), key=lambda item: median(item[1]))
    labels = [_short_label(label) for label, _ in ordered]
    data = [values for _, values in ordered]

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.6), 5.8))
    box = ax.boxplot(data, patch_artist=True, showfliers=False)

    for idx, patch in enumerate(box["boxes"]):
        patch.set_facecolor(PALETTE[idx % len(PALETTE)])
        patch.set_alpha(0.4)

    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xticks(list(range(1, len(labels) + 1)))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.25)

    _save_figure(fig, output_path)


def _plot_binned_trend(
    games: list[dict[str, Any]],
    group_key: str,
    metric_key: str,
    title: str,
    y_label: str,
    output_path: str,
) -> None:
    buckets: dict[str, list[float]] = defaultdict(list)

    for game in games:
        if group_key not in game or metric_key not in game:
            continue
        try:
            buckets[str(game[group_key])].append(float(game[metric_key]))
        except (TypeError, ValueError):
            continue

    if not buckets:
        return

    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    any_line = False

    for idx, (label, values) in enumerate(sorted(buckets.items(), key=lambda item: item[0])):
        xs, means, p25, p75, bin_size = _binned_stats(values)
        if not xs:
            continue

        color = PALETTE[idx % len(PALETTE)]
        display_label = f"{_short_label(label)} (bin={bin_size})"

        ax.plot(xs, means, color=color, linewidth=2.0, label=display_label)
        ax.fill_between(xs, p25, p75, color=color, alpha=0.18)
        any_line = True

    if not any_line:
        plt.close(fig)
        return

    ax.set_title(title)
    ax.set_xlabel("Game Number (within group)")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)

    _save_figure(fig, output_path)


def _plot_exp1(games: list[dict[str, Any]], summary: dict[str, Any], output_dir: str) -> list[str]:
    outputs: list[str] = []
    match_rows = summary.get("match_summaries", [])

    if isinstance(match_rows, list) and match_rows:
        out = os.path.join(output_dir, "readable_outcome_rates.png")
        _plot_stacked_rates(
            match_rows,
            label_key="matchup",
            title="Experiment 1: Win/Draw/Loss Rates by Matchup",
            output_path=out,
        )
        outputs.append(out)

    nodes_dist = os.path.join(output_dir, "readable_nodes_distribution.png")
    _plot_box_distribution(
        games,
        group_key="matchup",
        metric_key="avg_nodes_per_move",
        title="Experiment 1: Nodes/Move Distribution by Matchup",
        y_label="Avg Nodes per Move",
        output_path=nodes_dist,
    )
    if os.path.isfile(nodes_dist):
        outputs.append(nodes_dist)

    time_dist = os.path.join(output_dir, "readable_time_distribution.png")
    _plot_box_distribution(
        games,
        group_key="matchup",
        metric_key="avg_time_per_move_ms",
        title="Experiment 1: Time/Move Distribution by Matchup",
        y_label="Avg Time per Move (ms)",
        output_path=time_dist,
    )
    if os.path.isfile(time_dist):
        outputs.append(time_dist)

    nodes_trend = os.path.join(output_dir, "readable_nodes_binned_trend.png")
    _plot_binned_trend(
        games,
        group_key="matchup",
        metric_key="avg_nodes_per_move",
        title="Experiment 1: Smoothed Nodes/Move Trend by Matchup",
        y_label="Avg Nodes per Move",
        output_path=nodes_trend,
    )
    if os.path.isfile(nodes_trend):
        outputs.append(nodes_trend)

    time_trend = os.path.join(output_dir, "readable_time_binned_trend.png")
    _plot_binned_trend(
        games,
        group_key="matchup",
        metric_key="avg_time_per_move_ms",
        title="Experiment 1: Smoothed Time/Move Trend by Matchup",
        y_label="Avg Time per Move (ms)",
        output_path=time_trend,
    )
    if os.path.isfile(time_trend):
        outputs.append(time_trend)

    return outputs


def _plot_exp2(games: list[dict[str, Any]], summary: dict[str, Any], output_dir: str) -> list[str]:
    outputs: list[str] = []
    config_rows = summary.get("config_summaries", [])

    if isinstance(config_rows, list) and config_rows:
        out = os.path.join(output_dir, "readable_outcome_rates.png")
        _plot_stacked_rates(
            config_rows,
            label_key="configuration",
            title="Experiment 2: Win/Draw/Loss Rates by Configuration",
            output_path=out,
        )
        outputs.append(out)

    nodes_dist = os.path.join(output_dir, "readable_nodes_distribution.png")
    _plot_box_distribution(
        games,
        group_key="configuration",
        metric_key="avg_nodes_per_move",
        title="Experiment 2: Nodes/Move Distribution by Configuration",
        y_label="Avg Nodes per Move",
        output_path=nodes_dist,
    )
    if os.path.isfile(nodes_dist):
        outputs.append(nodes_dist)

    time_dist = os.path.join(output_dir, "readable_time_distribution.png")
    _plot_box_distribution(
        games,
        group_key="configuration",
        metric_key="avg_time_per_move_ms",
        title="Experiment 2: Time/Move Distribution by Configuration",
        y_label="Avg Time per Move (ms)",
        output_path=time_dist,
    )
    if os.path.isfile(time_dist):
        outputs.append(time_dist)

    nodes_trend = os.path.join(output_dir, "readable_nodes_binned_trend.png")
    _plot_binned_trend(
        games,
        group_key="configuration",
        metric_key="avg_nodes_per_move",
        title="Experiment 2: Smoothed Nodes/Move Trend by Configuration",
        y_label="Avg Nodes per Move",
        output_path=nodes_trend,
    )
    if os.path.isfile(nodes_trend):
        outputs.append(nodes_trend)

    time_trend = os.path.join(output_dir, "readable_time_binned_trend.png")
    _plot_binned_trend(
        games,
        group_key="configuration",
        metric_key="avg_time_per_move_ms",
        title="Experiment 2: Smoothed Time/Move Trend by Configuration",
        y_label="Avg Time per Move (ms)",
        output_path=time_trend,
    )
    if os.path.isfile(time_trend):
        outputs.append(time_trend)

    return outputs


def _plot_exp3(summary: dict[str, Any], output_dir: str) -> list[str]:
    # Disabled experiment 3 readable summary is not generated.
    return []


def _render_target(target: PlotTarget) -> list[str]:
    games = _read_json(target.games_path)
    summary = _read_json(target.summary_path)

    if not isinstance(games, list) or not isinstance(summary, dict):
        return []

    output_dir = os.path.join(target.images_dir, READABLE_SUBDIR)
    os.makedirs(output_dir, exist_ok=True)

    if "match_summaries" in summary:
        return _plot_exp1(games, summary, output_dir)

    if "config_summaries" in summary:
        return _plot_exp2(games, summary, output_dir)

    if "depth_config_summaries" in summary:
        return _plot_exp3(summary, output_dir)

    return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate readable graphs from saved experiment JSON files.")
    parser.add_argument(
        "--results-root",
        type=str,
        default=DEFAULT_RESULTS_ROOT,
        help="Root results directory containing experiment folders.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = _discover_targets(args.results_root)

    if not targets:
        print("No eligible experiment run folders found with both JSON files and an images folder.")
        return

    print(f"Found {len(targets)} run folder(s) to process.")
    total_outputs = 0

    for target in targets:
        outputs = _render_target(target)
        if outputs:
            print(f"\n{target.run_dir}")
            for output in outputs:
                print(f"  saved: {output}")
            total_outputs += len(outputs)

    print(f"\nDone. Generated {total_outputs} readable image(s).")


if __name__ == "__main__":
    main()
