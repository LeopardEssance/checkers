"""
Experiment 2: Ablation Study

Research question: What is the contribution of each enhancement component
to agent performance against a Random opponent?

Configurations (one component disabled at a time):
  - Full (TT+Killer+History)
  - No TT (Killer+History)
  - No Killer (TT+History)
  - No History (TT+Killer)
  - No Move Ordering (TT only)
  - Baseline (AB only)

Each configuration is played as RED and as BLACK.

Metrics:
  - Win/Loss/Draw counts and rates
  - Average nodes per move (mean ± std)
  - Average time per move in ms (mean ± std)

Output:
  results/experiment2_ablation/d{depth}_n{games}/experiment2_ablation.json
  results/experiment2_ablation/d{depth}_n{games}/experiment2_summary.json
  results/experiment2_ablation/experiment2_ablation.json
  results/experiment2_ablation/experiment2_summary.json

Run:
  python -m experiments.experiment2_ablation
"""

from __future__ import annotations

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

DEPTH = 4
NUM_GAMES = 10
TIME_LIMIT = 5.0
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

CONFIGS = [
    "Full (TT+Killer+History)",
    "No TT (Killer+History)",
    "No Killer (TT+History)",
    "No History (TT+Killer)",
    "No Move Ordering (TT only)",
    "Baseline (AB only)",
]

CONFIG_COLORS = {
    "Full (TT+Killer+History)": "#B22222",
    "No TT (Killer+History)": "#1f77b4",
    "No Killer (TT+History)": "#2E8B57",
    "No History (TT+Killer)": "#FF8C00",
    "No Move Ordering (TT only)": "#6A5ACD",
    "Baseline (AB only)": "#696969",
}

class _NoOpHistory(dict):
    def __getitem__(self, key):
        return 0

    def get(self, key, default=0):
        return 0

    def __setitem__(self, key, value):
        return None

    def clear(self):
        return None

def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0

def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_value = _mean(values)
    return (sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)) ** 0.5

def _pct(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0

def _slug(text: str) -> str:
    return (
        text.lower()
        .replace("+", "plus")
        .replace("(", "")
        .replace(")", "")
        .replace(" ", "_")
    )

@dataclass
class AblationGameRecord:
    game_id: int
    configuration: str
    ablation_side: str
    winner: str
    ablation_won: bool
    draw: bool
    draw_reason: str
    move_limit_hit: bool
    total_moves: int
    ablation_total_nodes: int
    random_total_nodes: int
    ablation_total_time_s: float
    random_total_time_s: float
    ablation_avg_nodes_per_move: float
    ablation_avg_time_per_move_ms: float
    random_avg_nodes_per_move: float
    random_avg_time_per_move_ms: float
    avg_nodes_per_move: float
    avg_time_per_move_ms: float

@dataclass
class AblationConfigSummary:
    configuration: str
    total_games: int
    wins: int
    losses: int
    draws: int
    win_rate_pct: float
    loss_rate_pct: float
    draw_rate_pct: float
    red_games: int
    black_games: int
    red_win_rate_pct: float
    black_win_rate_pct: float
    avg_nodes_per_move_mean: float
    avg_nodes_per_move_std: float
    avg_time_per_move_ms_mean: float
    avg_time_per_move_ms_std: float

@dataclass
class AblationExperimentSummary:
    total_configurations: int
    total_games: int
    depth: int
    games_per_side: int
    timestamp: str
    config_summaries: list[AblationConfigSummary]

def _build_ablation_agent(
    label: str,
    player: str,
    depth: int,
    time_limit: float,
    rng_seed: int | None,
    stochastic_tiebreak: bool,
):
    rng = random.Random(rng_seed) if rng_seed is not None else None
    tie_rng = rng if stochastic_tiebreak else None

    if label == "Full (TT+Killer+History)":
        return TranspositionAgent(player, depth=depth, time_limit=time_limit, use_tt=True, rng=tie_rng)

    if label == "No TT (Killer+History)":
        return MoveOrderingAgent(
            player,
            depth=depth,
            time_limit=time_limit,
            use_killer=True,
            use_history=True,
            rng=tie_rng,
        )

    if label == "No Killer (TT+History)":
        agent = TranspositionAgent(player, depth=depth, time_limit=time_limit, use_tt=True, rng=tie_rng)
        agent._store_killer = lambda ply, move: None
        return agent

    if label == "No History (TT+Killer)":
        agent = TranspositionAgent(player, depth=depth, time_limit=time_limit, use_tt=True, rng=tie_rng)
        agent.history_table = _NoOpHistory()
        return agent

    if label == "No Move Ordering (TT only)":
        agent = TranspositionAgent(player, depth=depth, time_limit=time_limit, use_tt=True, rng=tie_rng)
        agent._store_killer = lambda ply, move: None
        agent.history_table = _NoOpHistory()
        agent._order_moves = lambda moves, ply, tt_best: list(moves)
        return agent

    if label == "Baseline (AB only)":
        return BaselineAgent(player, depth=depth, time_limit=time_limit, rng=tie_rng)

    raise ValueError(f"Unsupported configuration: {label}")

def _summarize_config(label: str, records: list[AblationGameRecord]) -> AblationConfigSummary:
    wins = sum(1 for record in records if record.ablation_won)
    losses = sum(1 for record in records if not record.ablation_won and not record.draw)
    draws = sum(1 for record in records if record.draw)

    red_records = [record for record in records if record.ablation_side == RED]
    black_records = [record for record in records if record.ablation_side == BLACK]

    node_values = [record.ablation_avg_nodes_per_move for record in records]
    time_values = [record.ablation_avg_time_per_move_ms for record in records]

    return AblationConfigSummary(
        configuration=label,
        total_games=len(records),
        wins=wins,
        losses=losses,
        draws=draws,
        win_rate_pct=_pct(wins, len(records)),
        loss_rate_pct=_pct(losses, len(records)),
        draw_rate_pct=_pct(draws, len(records)),
        red_games=len(red_records),
        black_games=len(black_records),
        red_win_rate_pct=_pct(sum(1 for record in red_records if record.ablation_won), len(red_records)),
        black_win_rate_pct=_pct(sum(1 for record in black_records if record.ablation_won), len(black_records)),
        avg_nodes_per_move_mean=_mean(node_values),
        avg_nodes_per_move_std=_std(node_values),
        avg_time_per_move_ms_mean=_mean(time_values),
        avg_time_per_move_ms_std=_std(time_values),
    )

def run_experiment(
    num_games: int = NUM_GAMES,
    depth: int = DEPTH,
    time_limit: float = TIME_LIMIT,
    stochastic_tiebreak: bool = True,
    opening_random_plies: int = 0,
    seed: int | None = None,
    verbose: bool = False,
) -> tuple[list[AblationGameRecord], AblationExperimentSummary]:
    print("=" * 60)
    print("Experiment 2: Ablation Study")
    print(f"Depth={depth} | {num_games * 2} games per configuration")
    print("Opponent: Random")
    print(f"Stochastic tie-break: {'ON' if stochastic_tiebreak else 'OFF'}")
    print(f"Opening random plies: {opening_random_plies}")
    if seed is not None:
        print(f"Seed={seed}")
    print("=" * 60)

    all_records: list[AblationGameRecord] = []
    config_summaries: list[AblationConfigSummary] = []
    game_id = 0
    run_rng = random.Random(seed) if seed is not None else random.Random()

    for label in CONFIGS:
        print(f"\n{label} vs Random ...")
        config_records: list[AblationGameRecord] = []

        for ablation_side in [RED, BLACK]:
            random_side = BLACK if ablation_side == RED else RED

            for _ in range(num_games):
                game_seed = run_rng.randrange(2**32)
                ablation_agent = _build_ablation_agent(
                    label,
                    ablation_side,
                    depth,
                    time_limit,
                    rng_seed=game_seed ^ 0xA5A5A5A5,
                    stochastic_tiebreak=stochastic_tiebreak,
                )
                random_agent = RandomAgent(random_side, rng=random.Random(game_seed ^ 0x5A5A5A5A))
                opening_rng = random.Random(game_seed ^ 0xC3C3C3C3)

                if ablation_side == RED:
                    result = play_game(
                        ablation_agent,
                        random_agent,
                        tracked_player=RED,
                        opening_random_plies=opening_random_plies,
                        opening_rng=opening_rng,
                        verbose=verbose,
                    )
                else:
                    result = play_game(
                        random_agent,
                        ablation_agent,
                        tracked_player=BLACK,
                        opening_random_plies=opening_random_plies,
                        opening_rng=opening_rng,
                        verbose=verbose,
                    )

                rec = AblationGameRecord(
                    game_id=game_id,
                    configuration=label,
                    ablation_side=ablation_side,
                    winner=result.winner,
                    ablation_won=(result.winner == ablation_side),
                    draw=result.draw,
                    draw_reason=result.draw_reason,
                    move_limit_hit=result.move_limit_hit,
                    total_moves=result.total_moves,
                    ablation_total_nodes=(
                        sum(result.nodes_per_move_red)
                        if ablation_side == RED
                        else sum(result.nodes_per_move_black)
                    ),
                    random_total_nodes=(
                        sum(result.nodes_per_move_black)
                        if ablation_side == RED
                        else sum(result.nodes_per_move_red)
                    ),
                    ablation_total_time_s=(
                        sum(result.time_per_move_red_s)
                        if ablation_side == RED
                        else sum(result.time_per_move_black_s)
                    ),
                    random_total_time_s=(
                        sum(result.time_per_move_black_s)
                        if ablation_side == RED
                        else sum(result.time_per_move_red_s)
                    ),
                    ablation_avg_nodes_per_move=(
                        result.avg_nodes_per_move_red
                        if ablation_side == RED
                        else result.avg_nodes_per_move_black
                    ),
                    ablation_avg_time_per_move_ms=(
                        result.avg_time_per_move_red_s * 1000.0
                        if ablation_side == RED
                        else result.avg_time_per_move_black_s * 1000.0
                    ),
                    random_avg_nodes_per_move=(
                        result.avg_nodes_per_move_black
                        if ablation_side == RED
                        else result.avg_nodes_per_move_red
                    ),
                    random_avg_time_per_move_ms=(
                        result.avg_time_per_move_black_s * 1000.0
                        if ablation_side == RED
                        else result.avg_time_per_move_red_s * 1000.0
                    ),
                    avg_nodes_per_move=(
                        result.avg_nodes_per_move_red
                        if ablation_side == RED
                        else result.avg_nodes_per_move_black
                    ),
                    avg_time_per_move_ms=(
                        result.avg_time_per_move_red_s * 1000.0
                        if ablation_side == RED
                        else result.avg_time_per_move_black_s * 1000.0
                    ),
                )
                all_records.append(rec)
                config_records.append(rec)
                game_id += 1

                status = "WIN" if rec.ablation_won else ("DRAW" if rec.draw else "LOSS")
                side = "RED" if rec.ablation_side == RED else "BLK"
                print(
                    f"  Game {game_id:>3}  [{side}]  {status:4}  "
                    f"moves={rec.total_moves:>3}  "
                    f"{label} nodes/move={rec.ablation_avg_nodes_per_move:>8.1f}  "
                    f"ms/move={rec.ablation_avg_time_per_move_ms:>7.2f}"
                )

        summary = _summarize_config(label, config_records)
        config_summaries.append(summary)
        print(
            f"  {label}: W={summary.wins} L={summary.losses} D={summary.draws} "
            f"({summary.win_rate_pct:.1f}%) | "
            f"AvgNodes={summary.avg_nodes_per_move_mean:,.1f} | "
            f"AvgTime={summary.avg_time_per_move_ms_mean:.3f}ms"
        )

    print("\n" + "=" * 78)
    print(
        f"{'Configuration':<31} {'W-L-D':<13} {'Win%':>6} "
        f"{'Nodes/move (mean±sd)':>24} {'Time ms/move (mean±sd)':>28}"
    )
    print("-" * 78)
    for summary in sorted(config_summaries, key=lambda item: (-item.win_rate_pct, item.avg_nodes_per_move_mean)):
        wld = f"{summary.wins}-{summary.losses}-{summary.draws}"
        print(
            f"{summary.configuration:<31} "
            f"{wld:<13} "
            f"{summary.win_rate_pct:>5.1f}% "
            f"{summary.avg_nodes_per_move_mean:>8.1f}±{summary.avg_nodes_per_move_std:<7.1f} "
            f"{summary.avg_time_per_move_ms_mean:>9.2f}±{summary.avg_time_per_move_ms_std:<9.2f}"
        )

    experiment_summary = AblationExperimentSummary(
        total_configurations=len(CONFIGS),
        total_games=len(all_records),
        depth=depth,
        games_per_side=num_games,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        config_summaries=config_summaries,
    )
    return all_records, experiment_summary

def save_results(
    records: list[AblationGameRecord],
    summary: AblationExperimentSummary,
    results_dir: str = RESULTS_DIR,
) -> None:
    base_dir = os.path.join(results_dir, "experiment2_ablation")
    os.makedirs(base_dir, exist_ok=True)

    run_folder = f"d{summary.depth}_n{summary.games_per_side * 2}"
    run_dir = os.path.join(base_dir, run_folder)
    os.makedirs(run_dir, exist_ok=True)

    run_games_path = os.path.join(run_dir, "experiment2_ablation.json")
    run_summary_path = os.path.join(run_dir, "experiment2_summary.json")

    latest_games_path = os.path.join(base_dir, "experiment2_ablation.json")
    latest_summary_path = os.path.join(base_dir, "experiment2_summary.json")

    with open(run_games_path, "w", encoding="utf-8") as file_handle:
        json.dump([asdict(record) for record in records], file_handle, indent=2)

    with open(run_summary_path, "w", encoding="utf-8") as file_handle:
        json.dump(asdict(summary), file_handle, indent=2)

    with open(latest_games_path, "w", encoding="utf-8") as file_handle:
        json.dump([asdict(record) for record in records], file_handle, indent=2)

    with open(latest_summary_path, "w", encoding="utf-8") as file_handle:
        json.dump(asdict(summary), file_handle, indent=2)

    print("\nResults saved:")
    print(f"  {run_games_path}")
    print(f"  {run_summary_path}")
    print(f"  {latest_games_path}")
    print(f"  {latest_summary_path}")

def _save_graphs(
    records: list[AblationGameRecord],
    summary: AblationExperimentSummary,
    results_dir: str = RESULTS_DIR,
) -> list[str]:
    base_dir = os.path.join(results_dir, "experiment2_ablation")
    run_folder = f"d{summary.depth}_n{summary.games_per_side * 2}"
    images_dir = os.path.join(base_dir, run_folder, "images")
    os.makedirs(images_dir, exist_ok=True)

    labels = [item.configuration for item in summary.config_summaries]
    colors = [CONFIG_COLORS.get(item.configuration, "#0072B2") for item in summary.config_summaries]
    output_paths: list[str] = []

    # Win-rate comparison.
    win_rate_path = os.path.join(images_dir, "ablation_win_rate.png")
    fig, ax = plt.subplots(figsize=(11, 6))
    win_rates = [item.win_rate_pct for item in summary.config_summaries]
    x_values = list(range(len(labels)))
    bars = ax.bar(x_values, win_rates, color=colors)
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Win Rate (%)")
    ax.set_title("Experiment 2 Ablation: Win Rate vs Random")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(win_rate_path, dpi=180)
    plt.close(fig)
    output_paths.append(win_rate_path)

    # Nodes/move comparison.
    nodes_path = os.path.join(images_dir, "ablation_avg_nodes_per_move.png")
    fig, ax = plt.subplots(figsize=(11, 6))
    node_means = [item.avg_nodes_per_move_mean for item in summary.config_summaries]
    node_stds = [item.avg_nodes_per_move_std for item in summary.config_summaries]
    bars = ax.bar(x_values, node_means, yerr=node_stds, capsize=4, color=colors, alpha=0.95)
    ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Nodes / Move")
    ax.set_title("Experiment 2 Ablation: Search Cost")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(nodes_path, dpi=180)
    plt.close(fig)
    output_paths.append(nodes_path)

    # Time/move comparison.
    time_path = os.path.join(images_dir, "ablation_avg_time_per_move_ms.png")
    fig, ax = plt.subplots(figsize=(11, 6))
    time_means = [item.avg_time_per_move_ms_mean for item in summary.config_summaries]
    time_stds = [item.avg_time_per_move_ms_std for item in summary.config_summaries]
    bars = ax.bar(x_values, time_means, yerr=time_stds, capsize=4, color=colors, alpha=0.95)
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Milliseconds / Move")
    ax.set_title("Experiment 2 Ablation: Runtime Cost")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(time_path, dpi=180)
    plt.close(fig)
    output_paths.append(time_path)

    # Outcome breakdown comparison.
    outcomes_path = os.path.join(images_dir, "ablation_outcomes_stacked.png")
    fig, ax = plt.subplots(figsize=(11, 6))
    wins = [item.wins for item in summary.config_summaries]
    losses = [item.losses for item in summary.config_summaries]
    draws = [item.draws for item in summary.config_summaries]
    bars_wins = ax.bar(x_values, wins, label="Wins", color="#2E8B57")
    bars_losses = ax.bar(x_values, losses, bottom=wins, label="Losses", color="#B22222")
    bottoms = [wins[i] + losses[i] for i in range(len(wins))]
    bars_draws = ax.bar(x_values, draws, bottom=bottoms, label="Draws", color="#696969")
    ax.bar_label(bars_wins, fmt="%d", padding=2)
    ax.bar_label(bars_losses, fmt="%d", padding=2)
    ax.bar_label(bars_draws, fmt="%d", padding=2)
    max_height = max([wins[i] + losses[i] + draws[i] for i in range(len(wins))])
    ax.set_ylim(0, max_height * 1.15)
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Game Count")
    ax.set_title("Experiment 2 Ablation: Outcome Composition")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc='upper right')
    fig.subplots_adjust(right=0.80)
    fig.tight_layout()
    fig.savefig(outcomes_path, dpi=180)
    plt.close(fig)
    output_paths.append(outcomes_path)

    # Combined line chart: nodes per move by game for each configuration.
    nodes_line_path = os.path.join(images_dir, "ablation_nodes_per_move_line_combined.png")
    fig, ax = plt.subplots(figsize=(11, 6))
    for label in labels:
        config_records = [record for record in records if record.configuration == label]
        if not config_records:
            continue
        x_line = list(range(1, len(config_records) + 1))
        y_line = [record.ablation_avg_nodes_per_move for record in config_records]
        ax.plot(
            x_line,
            y_line,
            label=label,
            color=CONFIG_COLORS.get(label, "#0072B2"),
            linewidth=1.8,
            alpha=0.9,
        )
    ax.set_title("Experiment 2 Ablation: Nodes per Move by Game")
    ax.set_xlabel("Game # (per configuration)")
    ax.set_ylabel("Nodes / Move")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(nodes_line_path, dpi=180)
    plt.close(fig)
    output_paths.append(nodes_line_path)

    # Combined line chart: time per move by game for each configuration.
    time_line_path = os.path.join(images_dir, "ablation_time_per_move_line_combined.png")
    fig, ax = plt.subplots(figsize=(11, 6))
    for label in labels:
        config_records = [record for record in records if record.configuration == label]
        if not config_records:
            continue
        x_line = list(range(1, len(config_records) + 1))
        y_line = [record.ablation_avg_time_per_move_ms for record in config_records]
        ax.plot(
            x_line,
            y_line,
            label=label,
            color=CONFIG_COLORS.get(label, "#0072B2"),
            linewidth=1.8,
            alpha=0.9,
        )
    ax.set_title("Experiment 2 Ablation: Time per Move by Game")
    ax.set_xlabel("Game # (per configuration)")
    ax.set_ylabel("Milliseconds / Move")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(time_line_path, dpi=180)
    plt.close(fig)
    output_paths.append(time_line_path)

    return output_paths

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 2: Ablation Study")
    parser.add_argument("--games", type=int, default=NUM_GAMES, help="Games per side per configuration")
    parser.add_argument("--depth", type=int, default=DEPTH, help="Search depth for non-random agents")
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
    parser.add_argument("--verbose", action="store_true", help="Print board each move")
    parser.add_argument("--no-save", action="store_true", help="Skip writing results to disk")
    parser.add_argument("--no-plot", action="store_true", help="Skip graph generation")
    args = parser.parse_args()

    records, summary = run_experiment(
        num_games=args.games,
        depth=args.depth,
        time_limit=args.time_limit,
        stochastic_tiebreak=args.stochastic_tiebreak,
        opening_random_plies=max(0, args.opening_random_plies),
        seed=args.seed,
        verbose=args.verbose,
    )

    if not args.no_save:
        save_results(records, summary)

    if args.no_plot:
        raise SystemExit(0)

    graph_paths = _save_graphs(records, summary)
    print("\nSaved graphs:")
    for graph_path in graph_paths:
        print(f"  {graph_path}")
