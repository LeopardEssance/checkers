"""
Experiment 3: Scalability - Search Depth vs Performance

Research question: How does each configuration scale as search depth changes?

Depths tested: 3, 4, 5, 6
Configs:       Baseline, MoveOrdering, Enhanced
Metrics:       win rate, average nodes/move, average time/move

Output:
  results/experiment3_scalability/d{depth_span}_n{games}/experiment3_scalability.json
  results/experiment3_scalability/d{depth_span}_n{games}/experiment3_summary.json
  results/experiment3_scalability/experiment3_scalability.json
  results/experiment3_scalability/experiment3_summary.json

Run:
  python -m experiments.experiment3_scalability
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import random
import time
from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt

from src.agents.base import BaselineAgent, RandomAgent
from src.agents.move_ordering import MoveOrderingAgent
from src.agents.transposition import TranspositionAgent
from src.engine.board import BLACK, RED
from src.engine.game_runner import play_game

DEPTHS = [3, 4, 5, 6]
CONFIGS = ["Baseline", "MoveOrdering", "Enhanced"]
NUM_GAMES = 5
TIME_LIMIT = 5.0
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

CONFIG_COLORS = {
    "Baseline": "#1f77b4",
    "MoveOrdering": "#2E8B57",
    "Enhanced": "#B22222",
}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_value = _mean(values)
    return (sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def _pct(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


@dataclass
class ScalabilityGameRecord:
    game_id: int
    depth: int
    configuration: str
    config_side: str
    winner: str
    config_won: bool
    draw: bool
    draw_reason: str
    move_limit_hit: bool
    total_moves: int
    config_avg_nodes_per_move: float
    config_avg_time_per_move_ms: float
    random_avg_nodes_per_move: float
    random_avg_time_per_move_ms: float


@dataclass
class ScalabilityDepthConfigSummary:
    depth: int
    configuration: str
    total_games: int
    wins: int
    losses: int
    draws: int
    win_rate_pct: float
    avg_nodes_per_move_mean: float
    avg_nodes_per_move_std: float
    avg_time_per_move_ms_mean: float
    avg_time_per_move_ms_std: float


@dataclass
class ScalabilityExperimentSummary:
    depths: list[int]
    configurations: list[str]
    total_games: int
    games_per_side: int
    timestamp: str
    depth_config_summaries: list[ScalabilityDepthConfigSummary]


@dataclass(frozen=True)
class ScalabilityGameJob:
    game_id: int
    depth: int
    configuration: str
    config_side: str
    time_limit: float
    stochastic_tiebreak: bool
    opening_random_plies: int
    game_seed: int
    verbose: bool


def _build_scalability_agent(
    name: str,
    player: str,
    depth: int,
    time_limit: float,
    rng_seed: int | None,
    stochastic_tiebreak: bool,
):
    rng = random.Random(rng_seed) if rng_seed is not None else None
    tie_rng = rng if stochastic_tiebreak else None

    if name == "Baseline":
        return BaselineAgent(player, depth=depth, time_limit=time_limit, rng=tie_rng)

    if name == "MoveOrdering":
        return MoveOrderingAgent(
            player,
            depth=depth,
            time_limit=time_limit,
            use_killer=True,
            use_history=True,
            rng=tie_rng,
        )

    if name == "Enhanced":
        return TranspositionAgent(player, depth=depth, time_limit=time_limit, use_tt=True, rng=tie_rng)

    raise ValueError(f"Unsupported configuration: {name}")


def _summarize_depth_config(
    depth: int,
    configuration: str,
    records: list[ScalabilityGameRecord],
) -> ScalabilityDepthConfigSummary:
    wins = sum(1 for record in records if record.config_won)
    losses = sum(1 for record in records if not record.config_won and not record.draw)
    draws = sum(1 for record in records if record.draw)
    node_values = [record.config_avg_nodes_per_move for record in records]
    time_values = [record.config_avg_time_per_move_ms for record in records]

    return ScalabilityDepthConfigSummary(
        depth=depth,
        configuration=configuration,
        total_games=len(records),
        wins=wins,
        losses=losses,
        draws=draws,
        win_rate_pct=_pct(wins, len(records)),
        avg_nodes_per_move_mean=_mean(node_values),
        avg_nodes_per_move_std=_std(node_values),
        avg_time_per_move_ms_mean=_mean(time_values),
        avg_time_per_move_ms_std=_std(time_values),
    )


def _run_scalability_game_job(job: ScalabilityGameJob) -> ScalabilityGameRecord:
    config_side = job.config_side
    random_side = BLACK if config_side == RED else RED

    agent_config = _build_scalability_agent(
        job.configuration,
        config_side,
        job.depth,
        job.time_limit,
        rng_seed=job.game_seed ^ 0xA5A5A5A5,
        stochastic_tiebreak=job.stochastic_tiebreak,
    )
    agent_random = RandomAgent(random_side, rng=random.Random(job.game_seed ^ 0x5A5A5A5A))
    opening_rng = random.Random(job.game_seed ^ 0xC3C3C3C3)

    if config_side == RED:
        result = play_game(
            agent_config,
            agent_random,
            tracked_player=RED,
            opening_random_plies=job.opening_random_plies,
            opening_rng=opening_rng,
            verbose=job.verbose,
        )
    else:
        result = play_game(
            agent_random,
            agent_config,
            tracked_player=BLACK,
            opening_random_plies=job.opening_random_plies,
            opening_rng=opening_rng,
            verbose=job.verbose,
        )

    return ScalabilityGameRecord(
        game_id=job.game_id,
        depth=job.depth,
        configuration=job.configuration,
        config_side=config_side,
        winner=result.winner,
        config_won=(result.winner == config_side),
        draw=result.draw,
        draw_reason=result.draw_reason,
        move_limit_hit=result.move_limit_hit,
        total_moves=result.total_moves,
        config_avg_nodes_per_move=(
            result.avg_nodes_per_move_red if config_side == RED else result.avg_nodes_per_move_black
        ),
        config_avg_time_per_move_ms=(
            result.avg_time_per_move_red_s * 1000.0
            if config_side == RED
            else result.avg_time_per_move_black_s * 1000.0
        ),
        random_avg_nodes_per_move=(
            result.avg_nodes_per_move_black if config_side == RED else result.avg_nodes_per_move_red
        ),
        random_avg_time_per_move_ms=(
            result.avg_time_per_move_black_s * 1000.0
            if config_side == RED
            else result.avg_time_per_move_red_s * 1000.0
        ),
    )


def run_experiment(
    games_per_side: int = NUM_GAMES,
    depths: list[int] | None = None,
    time_limit: float = TIME_LIMIT,
    stochastic_tiebreak: bool = True,
    opening_random_plies: int = 0,
    seed: int | None = None,
    max_workers: int | None = None,
    verbose: bool = False,
    show_progress: bool = True,
) -> tuple[list[ScalabilityGameRecord], ScalabilityExperimentSummary]:
    if depths is None:
        depths = DEPTHS

    depths = sorted(set(depths))

    if show_progress:
        print("=" * 72)
        print("Experiment 3: Scalability - Depth vs Performance")
        print(f"Depths={depths} | {games_per_side * 2} games per config-depth")
        print("Opponent: Random")
        print(f"Stochastic tie-break: {'ON' if stochastic_tiebreak else 'OFF'}")
        print(f"Opening random plies: {opening_random_plies}")
        if seed is not None:
            print(f"Seed={seed}")
        print("=" * 72)

    all_records: list[ScalabilityGameRecord] = []
    summaries: list[ScalabilityDepthConfigSummary] = []
    game_id = 0
    run_rng = random.Random(seed) if seed is not None else random.Random()
    parallel_enabled = not verbose and (max_workers is None or max_workers > 1)

    if max_workers is not None and max_workers < 1:
        raise ValueError("max_workers must be >= 1 or None")

    if show_progress and parallel_enabled:
        worker_label = "default" if max_workers is None else str(max_workers)
        print(f"Parallel game execution enabled (workers={worker_label})")

    executor_cm = (
        concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
        if parallel_enabled
        else None
    )

    if executor_cm is None:
        executor_context = None
    else:
        executor_context = executor_cm.__enter__()

    try:
        for depth in depths:
            if show_progress:
                print(f"\nDepth {depth} ...")
            for configuration in CONFIGS:
                if show_progress:
                    print(f"  {configuration} vs Random")
                bucket: list[ScalabilityGameRecord] = []
                jobs: list[ScalabilityGameJob] = []

                for config_side in [RED, BLACK]:
                    for _ in range(games_per_side):
                        jobs.append(
                            ScalabilityGameJob(
                                game_id=game_id,
                                depth=depth,
                                configuration=configuration,
                                config_side=config_side,
                                time_limit=time_limit,
                                stochastic_tiebreak=stochastic_tiebreak,
                                opening_random_plies=opening_random_plies,
                                game_seed=run_rng.randrange(2**32),
                                verbose=verbose,
                            )
                        )
                        game_id += 1

                if executor_context is None:
                    records = [_run_scalability_game_job(job) for job in jobs]
                else:
                    records = list(executor_context.map(_run_scalability_game_job, jobs))

                for record in records:
                    all_records.append(record)
                    bucket.append(record)

                    if show_progress:
                        status = "WIN" if record.config_won else ("DRAW" if record.draw else "LOSS")
                        side = "RED" if record.config_side == RED else "BLK"
                        print(
                            f"    Game {record.game_id + 1:>4}  [{side}]  {status:4}  "
                            f"moves={record.total_moves:>3}  "
                            f"{configuration} nodes/move={record.config_avg_nodes_per_move:>8.1f}  "
                            f"ms/move={record.config_avg_time_per_move_ms:>7.2f}"
                        )

                summary = _summarize_depth_config(depth, configuration, bucket)
                summaries.append(summary)
                if show_progress:
                    print(
                        f"    W={summary.wins} L={summary.losses} D={summary.draws} "
                        f"({summary.win_rate_pct:.1f}%) | "
                        f"AvgNodes={summary.avg_nodes_per_move_mean:,.1f} | "
                        f"AvgTime={summary.avg_time_per_move_ms_mean:.3f}ms"
                    )
    finally:
        if executor_cm is not None:
            executor_cm.__exit__(None, None, None)

    if show_progress:
        print("\n" + "=" * 92)
        print(
            f"{'Depth':<7} {'Configuration':<13} {'W-L-D':<12} {'Win%':>6} "
            f"{'Nodes/move (mean+-sd)':>24} {'Time ms/move (mean+-sd)':>29}"
        )
        print("-" * 92)
        for summary in sorted(summaries, key=lambda item: (item.depth, item.configuration)):
            wld = f"{summary.wins}-{summary.losses}-{summary.draws}"
            print(
                f"{summary.depth:<7} "
                f"{summary.configuration:<13} "
                f"{wld:<12} "
                f"{summary.win_rate_pct:>5.1f}% "
                f"{summary.avg_nodes_per_move_mean:>8.1f}+-{summary.avg_nodes_per_move_std:<8.1f} "
                f"{summary.avg_time_per_move_ms_mean:>9.2f}+-{summary.avg_time_per_move_ms_std:<9.2f}"
            )

    experiment_summary = ScalabilityExperimentSummary(
        depths=depths,
        configurations=CONFIGS,
        total_games=len(all_records),
        games_per_side=games_per_side,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        depth_config_summaries=summaries,
    )

    return all_records, experiment_summary


def save_results(
    records: list[ScalabilityGameRecord],
    summary: ScalabilityExperimentSummary,
    results_dir: str = RESULTS_DIR,
    show_progress: bool = True,
) -> None:
    base_dir = os.path.join(results_dir, "experiment3_scalability")
    os.makedirs(base_dir, exist_ok=True)

    depth_span = f"{min(summary.depths)}to{max(summary.depths)}"
    run_folder = f"d{depth_span}_n{summary.games_per_side * 2}"
    run_dir = os.path.join(base_dir, run_folder)
    os.makedirs(run_dir, exist_ok=True)

    run_games_path = os.path.join(run_dir, "experiment3_scalability.json")
    run_summary_path = os.path.join(run_dir, "experiment3_summary.json")
    latest_games_path = os.path.join(base_dir, "experiment3_scalability.json")
    latest_summary_path = os.path.join(base_dir, "experiment3_summary.json")

    with open(run_games_path, "w", encoding="utf-8") as file_handle:
        json.dump([asdict(record) for record in records], file_handle, indent=2)

    with open(run_summary_path, "w", encoding="utf-8") as file_handle:
        json.dump(asdict(summary), file_handle, indent=2)

    with open(latest_games_path, "w", encoding="utf-8") as file_handle:
        json.dump([asdict(record) for record in records], file_handle, indent=2)

    with open(latest_summary_path, "w", encoding="utf-8") as file_handle:
        json.dump(asdict(summary), file_handle, indent=2)

    if show_progress:
        print("\nResults saved:")
        print(f"  {run_games_path}")
        print(f"  {run_summary_path}")
        print(f"  {latest_games_path}")
        print(f"  {latest_summary_path}")


def _save_graphs(
    summary: ScalabilityExperimentSummary,
    results_dir: str = RESULTS_DIR,
    output_images_dir: str | None = None,
) -> list[str]:
    """
    Generate comparison graphs showing how each configuration scales across depths.
    
    Produces three charts:
      1. Win-rate bar chart (grouped by depth, configurations as bars)
      2. Nodes-per-move line chart with error bars (configuration lines)
      3. Time-per-move line chart with error bars (configuration lines)
    """
    if output_images_dir is None:
        base_dir = os.path.join(results_dir, "experiment3_scalability")
        depth_span = f"{min(summary.depths)}to{max(summary.depths)}"
        run_folder = f"d{depth_span}_n{summary.games_per_side * 2}"
        images_dir = os.path.join(base_dir, run_folder, "images")
    else:
        images_dir = output_images_dir

    os.makedirs(images_dir, exist_ok=True)

    output_paths: list[str] = []

    # Organize summaries by configuration (each configuration has multiple depths)
    by_config: dict[str, list[ScalabilityDepthConfigSummary]] = {config: [] for config in summary.configurations}
    for item in summary.depth_config_summaries:
        by_config.setdefault(item.configuration, []).append(item)

    # Sort each configuration's depths in ascending order
    for config in by_config:
        by_config[config] = sorted(by_config[config], key=lambda x: x.depth)

    # === Win-rate comparison: grouped bars (one group per depth) ===
    win_path = os.path.join(images_dir, "scalability_compare_win_rate_by_depth.png")
    fig, ax = plt.subplots(figsize=(10, 6))
    depths_sorted = sorted(summary.depths)
    x_positions = list(range(len(depths_sorted)))
    num_configs = max(1, len(summary.configurations))
    bar_width = min(0.8 / num_configs, 0.28)
    start_offset = -0.5 * bar_width * (num_configs - 1)

    for idx, configuration in enumerate(summary.configurations):
        points = by_config.get(configuration, [])
        rate_by_depth = {point.depth: point.win_rate_pct for point in points}
        y_values = [rate_by_depth.get(depth, 0.0) for depth in depths_sorted]
        x_values = [x + start_offset + idx * bar_width for x in x_positions]

        ax.bar(
            x_values,
            y_values,
            width=bar_width,
            label=configuration,
            color=CONFIG_COLORS.get(configuration, "#0072B2"),
            alpha=0.9,
            edgecolor="white",
            linewidth=0.6,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(depths_sorted)
    ax.set_title("Experiment 3: Win Rate by Depth")
    ax.set_xlabel("Depth")
    ax.set_ylabel("Win Rate (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(win_path, dpi=180)
    plt.close(fig)
    output_paths.append(win_path)

    # === Nodes-per-move: line plot with error bars ===
    nodes_path = os.path.join(images_dir, "scalability_compare_nodes_by_depth.png")
    fig, ax = plt.subplots(figsize=(10, 6))
    for configuration, points in by_config.items():
        if not points:
            continue
        x_values = [point.depth for point in points]
        means = [point.avg_nodes_per_move_mean for point in points]
        stds = [point.avg_nodes_per_move_std for point in points]
        ax.errorbar(
            x_values,
            means,
            yerr=stds,
            marker="o",
            capsize=4,
            label=configuration,
            color=CONFIG_COLORS.get(configuration, "#0072B2"),
            linewidth=2.0,
            alpha=0.95,
        )
    ax.set_title("Experiment 3: Nodes per Move by Depth")
    ax.set_xlabel("Depth")
    ax.set_ylabel("Nodes / Move (mean +/- sd)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(nodes_path, dpi=180)
    plt.close(fig)
    output_paths.append(nodes_path)

    # === Time-per-move: line plot with error bars ===
    time_path = os.path.join(images_dir, "scalability_compare_time_by_depth_ms.png")
    fig, ax = plt.subplots(figsize=(10, 6))
    for configuration, points in by_config.items():
        if not points:
            continue
        x_values = [point.depth for point in points]
        means = [point.avg_time_per_move_ms_mean for point in points]
        stds = [point.avg_time_per_move_ms_std for point in points]
        ax.errorbar(
            x_values,
            means,
            yerr=stds,
            marker="o",
            capsize=4,
            label=configuration,
            color=CONFIG_COLORS.get(configuration, "#0072B2"),
            linewidth=2.0,
            alpha=0.95,
        )
    ax.set_title("Experiment 3: Time per Move by Depth")
    ax.set_xlabel("Depth")
    ax.set_ylabel("Milliseconds / Move (mean +/- sd)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(time_path, dpi=180)
    plt.close(fig)
    output_paths.append(time_path)

    return output_paths


def _load_summary_from_json(summary_path: str) -> ScalabilityExperimentSummary:
    with open(summary_path, "r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)

    rows = [
        ScalabilityDepthConfigSummary(
            depth=int(item["depth"]),
            configuration=str(item["configuration"]),
            total_games=int(item["total_games"]),
            wins=int(item["wins"]),
            losses=int(item["losses"]),
            draws=int(item["draws"]),
            win_rate_pct=float(item["win_rate_pct"]),
            avg_nodes_per_move_mean=float(item["avg_nodes_per_move_mean"]),
            avg_nodes_per_move_std=float(item["avg_nodes_per_move_std"]),
            avg_time_per_move_ms_mean=float(item["avg_time_per_move_ms_mean"]),
            avg_time_per_move_ms_std=float(item["avg_time_per_move_ms_std"]),
        )
        for item in payload.get("depth_config_summaries", [])
    ]

    return ScalabilityExperimentSummary(
        depths=[int(depth) for depth in payload.get("depths", [])],
        configurations=[str(config) for config in payload.get("configurations", [])],
        total_games=int(payload.get("total_games", 0)),
        games_per_side=int(payload.get("games_per_side", 0)),
        timestamp=str(payload.get("timestamp", "")),
        depth_config_summaries=rows,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 3: Scalability - Depth vs Performance")
    parser.add_argument("--games", type=int, default=NUM_GAMES, help="Games per side per depth/config")
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=DEPTHS,
        help="Depth values to evaluate, for example: --depths 3 4 5 6",
    )
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT, help="Time limit per move (s)")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducible stochastic runs")
    parser.add_argument(
        "--stochastic-tiebreak",
        action="store_true",
        help="Randomly break near-equal best-move ties in search agents",
    )
    parser.add_argument(
        "--opening-random-plies",
        type=int,
        default=0,
        help="Number of opening plies chosen randomly before normal search",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Process workers for parallel game execution (default: CPU count). Use 1 for serial mode.",
    )
    parser.add_argument(
        "--plot-from-run-dir",
        type=str,
        default=None,
        help="Reuse an existing run directory and regenerate graphs from experiment3_summary.json without running games",
    )
    parser.add_argument("--verbose", action="store_true", help="Print board each move")
    parser.add_argument(
        "--done-only",
        action="store_true",
        help="Suppress progress output and print only a final Done line",
    )
    parser.add_argument("--no-save", action="store_true", help="Skip writing results to disk")
    parser.add_argument("--no-plot", action="store_true", help="Skip graph generation")
    args = parser.parse_args()
    show_progress = not args.done_only

    if args.plot_from_run_dir:
        summary_path = os.path.join(args.plot_from_run_dir, "experiment3_summary.json")
        if not os.path.isfile(summary_path):
            raise FileNotFoundError(f"Missing summary file: {summary_path}")

        summary = _load_summary_from_json(summary_path)
        graph_paths = _save_graphs(
            summary,
            output_images_dir=os.path.join(args.plot_from_run_dir, "images"),
        )
        if show_progress:
            print("\nSaved graphs from existing run:")
            for graph_path in graph_paths:
                print(f"  {graph_path}")
        print("Done")
        raise SystemExit(0)

    records, summary = run_experiment(
        games_per_side=max(1, args.games),
        depths=args.depths,
        time_limit=args.time_limit,
        stochastic_tiebreak=args.stochastic_tiebreak,
        opening_random_plies=max(0, args.opening_random_plies),
        seed=args.seed,
        max_workers=args.workers,
        verbose=args.verbose,
        show_progress=show_progress,
    )

    if not args.no_save:
        save_results(records, summary, show_progress=show_progress)

    if args.no_plot:
        print("Done")
        raise SystemExit(0)

    graph_paths = _save_graphs(summary)
    if show_progress:
        print("\nSaved graphs:")
        for graph_path in graph_paths:
            print(f"  {graph_path}")
    print("Done")
