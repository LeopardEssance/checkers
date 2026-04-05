"""
Generate human-readable performance graphs from experiment JSON outputs.

Default usage (baseline files):
    python -m experiments.plot_performance

Custom usage:
    python -m experiments.plot_performance \
        --dataset Baseline results/exp0_baseline_vs_random.json results/exp0_summary.json \
        --output-dir results/images

The --dataset flag can be provided multiple times to compare agents.
Each --dataset takes 3 arguments:
    LABEL GAMES_JSON_PATH SUMMARY_JSON_PATH
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt

DEFAULT_GAMES_PATH = os.path.join("results", "exp0_baseline_vs_random.json")
DEFAULT_SUMMARY_PATH = os.path.join("results", "exp0_summary.json")
DEFAULT_IMAGES_DIR = os.path.join("results", "images")

@dataclass
class ExperimentDataset:
    label: str
    games: list[dict[str, Any]]
    summary: dict[str, Any]


# Colorblind-friendly palette.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]

def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _infer_experiment_prefix(games_path: str, fallback_label: str) -> str:
    base = os.path.basename(games_path)
    match = re.match(r"(exp\d+)", base.lower())
    if match:
        return match.group(1)
    return _slug(fallback_label) or "experiment"


def _infer_run_folder_name(prefix: str, summary: dict[str, Any]) -> str:
    depth = summary.get("baseline_depth")
    total_games = summary.get("total_games", summary.get("num_games"))
    if depth is None or total_games is None:
        return prefix
    return f"{prefix}_d{depth}_n{total_games}"


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _load_dataset(label: str, games_path: str, summary_path: str) -> ExperimentDataset:
    games_data = _read_json(games_path)
    summary_data = _read_json(summary_path)

    if not isinstance(games_data, list):
        raise ValueError(f"Expected list in games file: {games_path}")
    if not isinstance(summary_data, dict):
        raise ValueError(f"Expected object in summary file: {summary_path}")

    return ExperimentDataset(label=label, games=games_data, summary=summary_data)

def _extract_series(games: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for g in games:
        raw = g.get(key)
        if raw is None:
            continue
        values.append(float(raw))
    return values

def _dataset_meta(dataset: ExperimentDataset) -> tuple[str, str]:
    depth = dataset.summary.get("baseline_depth", "?")
    total_games = dataset.summary.get("total_games", dataset.summary.get("num_games", "?"))
    return str(depth), str(total_games)

def _label_with_meta(dataset: ExperimentDataset, multiline: bool = False) -> str:
    depth, total_games = _dataset_meta(dataset)
    if multiline:
        return f"{dataset.label}\n(d={depth}, n={total_games})"
    return f"{dataset.label} (d={depth}, n={total_games})"

def _plot_outcomes(ax: plt.Axes, datasets: list[ExperimentDataset]) -> None:
    x_positions = list(range(len(datasets)))

    win_vals = [float(d.summary.get("wins", 0)) for d in datasets]
    loss_vals = [float(d.summary.get("losses", 0)) for d in datasets]
    draw_vals = [float(d.summary.get("draws", 0)) for d in datasets]

    width = 0.25
    ax.bar([x - width for x in x_positions], win_vals, width=width, label="Wins", color="#2E8B57")
    ax.bar(x_positions, loss_vals, width=width, label="Losses", color="#B22222")
    ax.bar([x + width for x in x_positions], draw_vals, width=width, label="Draws", color="#696969")

    ax.set_xticks(x_positions)
    ax.set_xticklabels([_label_with_meta(d, multiline=True) for d in datasets], rotation=0)
    ax.set_title("Outcomes")
    ax.set_ylabel("Game Count")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

def _plot_metric_by_game(ax: plt.Axes, datasets: list[ExperimentDataset], game_key: str, title: str, y_label: str) -> None:
    for idx, dataset in enumerate(datasets):
        series = _extract_series(dataset.games, game_key)
        if not series:
            continue
        x = list(range(1, len(series) + 1))
        ax.plot(
            x,
            series,
            label=_label_with_meta(dataset),
            color=PALETTE[idx % len(PALETTE)],
            linewidth=1.6,
            alpha=0.9,
        )

    ax.set_title(title)
    ax.set_xlabel("Game #")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

def _save_outcomes_chart(datasets: list[ExperimentDataset], output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_outcomes(ax, datasets)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

def _save_nodes_chart(datasets: list[ExperimentDataset], output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_metric_by_game(
        ax,
        datasets,
        game_key="avg_nodes_per_move",
        title="Average Nodes Expanded per Move (by Game)",
        y_label="Nodes / Move",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

def _save_time_chart(datasets: list[ExperimentDataset], output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_metric_by_game(
        ax,
        datasets,
        game_key="avg_time_per_move_ms",
        title="Average Time per Move (ms) (by Game)",
        y_label="Milliseconds / Move",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

def build_graph_images(datasets: list[ExperimentDataset], output_dir: str, file_prefix: str) -> list[str]:
    if not datasets:
        raise ValueError("No datasets provided")

    base = file_prefix if file_prefix else "comparison"
    experiment_output_dir = os.path.join(output_dir, base)
    os.makedirs(experiment_output_dir, exist_ok=True)

    outcomes_path = os.path.join(experiment_output_dir, f"{base}_winrate_outcomes.png")
    nodes_path = os.path.join(experiment_output_dir, f"{base}_avg_nodes_per_move.png")
    time_path = os.path.join(experiment_output_dir, f"{base}_avg_time_per_move.png")

    _save_outcomes_chart(datasets, outcomes_path)
    _save_nodes_chart(datasets, nodes_path)
    _save_time_chart(datasets, time_path)

    return [outcomes_path, nodes_path, time_path]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create experiment performance graphs")
    parser.add_argument(
        "--dataset",
        action="append",
        nargs=3,
        metavar=("LABEL", "GAMES_JSON", "SUMMARY_JSON"),
        help="Add a dataset for plotting. Can be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_IMAGES_DIR,
        help="Root directory where experiment subfolders are created for graph images.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    datasets: list[ExperimentDataset] = []
    source_games_path = DEFAULT_GAMES_PATH

    if args.dataset:
        for idx, (label, games_path, summary_path) in enumerate(args.dataset):
            if idx == 0:
                source_games_path = games_path
            datasets.append(_load_dataset(label, games_path, summary_path))
    else:
        datasets.append(_load_dataset("Baseline", DEFAULT_GAMES_PATH, DEFAULT_SUMMARY_PATH))

    if len(datasets) == 1:
        experiment_prefix = _infer_experiment_prefix(source_games_path, datasets[0].label)
        prefix = _infer_run_folder_name(experiment_prefix, datasets[0].summary)
    else:
        prefix = "comparison"

    output_paths = build_graph_images(datasets, args.output_dir, prefix)
    print("Saved performance graphs:")
    for path in output_paths:
        print(f"  {path}")

if __name__ == "__main__":
    main()
