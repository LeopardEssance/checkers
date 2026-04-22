"""
Experiment 4: Baseline vs No-Pruning

Research question: How much does alpha-beta pruning reduce computational cost
compared to plain minimax search?

I forgot about this one, otherwise it'd be an earlier experiment.. Whoops.

Configurations:
    - Baseline (Alpha-Beta)
    - No Pruning (Plain Minimax)

Depths tested:
    - 2, 3, 4

Opponent:
    - Random

Metrics:
    - Win rate
    - Average nodes per move (mean +/- std)
    - Average time per move in ms (mean +/- std)
    - Pruning efficiency (% nodes saved)

Output:
    results/experiment4_baseline_vs_nopruning/d{depth_span}_n{games}/experiment4_baseline_vs_nopruning.json
    results/experiment4_baseline_vs_nopruning/d{depth_span}_n{games}/experiment4_summary.json
    results/experiment4_baseline_vs_nopruning/experiment4_baseline_vs_nopruning.json
    results/experiment4_baseline_vs_nopruning/experiment4_summary.json

Run:
    python -m experiments.expiriment4_baseline_vs_nopruning
    python -m experiments.expiriment4_baseline_vs_nopruning --games 10 --depths 2 3 4
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from typing import Optional

import matplotlib.pyplot as plt

from src.agents.base import BaselineAgent, RandomAgent, evaluate
from src.engine.board import BLACK, RED, Board, LOSS_SCORE, Move, WIN_SCORE
from src.engine.game_runner import play_game

DEPTHS = [2, 3, 4]
NUM_GAMES = 10
TIME_LIMIT = 10.0  # Longer time limit since NoPruning is slower
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
CONFIGS = ["Baseline (Alpha-Beta)", "No Pruning (Plain Minimax)"]

CONFIG_COLORS = {
    "Baseline (Alpha-Beta)": "#2E8B57",
    "No Pruning (Plain Minimax)": "#B22222",
}

class NoPruningAgent:
    """Plain minimax search with the same evaluation as the baseline agent."""

    def __init__(
        self,
        player: str,
        depth: int = 5,
        time_limit: float = 5.0,
        rng: Optional[random.Random] = None,
        tie_break_eps: float = 1e-9,
    ):
        self.player = player
        self.depth = depth
        self.time_limit = time_limit
        self.rng = rng
        self.tie_break_eps = tie_break_eps

        self.nodes_expanded = 0
        self.start_time = 0.0

    def choose_move(self, board: Board) -> Optional[Move]:
        legal_moves = board.get_legal_moves(self.player)
        if not legal_moves:
            return None

        self.nodes_expanded = 0
        self.start_time = time.time()

        best_score = float("-inf")
        best_moves: list[Move] = []

        for move in legal_moves:
            score = self._minimax(
                board.apply_move(move),
                depth=self.depth - 1,
                maximizing=False,
            )
            if score > best_score + self.tie_break_eps:
                best_score = score
                best_moves = [move]
            elif abs(score - best_score) <= self.tie_break_eps:
                best_moves.append(move)

        if not best_moves:
            return None
        if self.rng is not None:
            return self.rng.choice(best_moves)
        return best_moves[0]

    def _minimax(self, board: Board, depth: int, maximizing: bool) -> float:
        self.nodes_expanded += 1

        if time.time() - self.start_time >= self.time_limit:
            return evaluate(board, self.player)

        terminal_val = board.terminal_score(self.player)
        if terminal_val is not None:
            return terminal_val

        if depth == 0:
            return evaluate(board, self.player)

        current_player = board.current_player
        legal_moves = board.get_legal_moves(current_player)
        if not legal_moves:
            return LOSS_SCORE if current_player == self.player else WIN_SCORE

        if maximizing:
            value = float("-inf")
            for move in legal_moves:
                value = max(value, self._minimax(board.apply_move(move), depth - 1, False))
            return value

        value = float("inf")
        for move in legal_moves:
            value = min(value, self._minimax(board.apply_move(move), depth - 1, True))
        return value

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
class GameRecord:
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
class DepthConfigSummary:
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
class ExperimentSummary:
    depths: list[int]
    configurations: list[str]
    total_games: int
    games_per_side: int
    timestamp: str
    depth_config_summaries: list[DepthConfigSummary]

    baseline_avg_nodes: float
    nopruning_avg_nodes: float
    pruning_efficiency_pct: float

def _build_agent(
    config_name: str,
    player: str,
    depth: int,
    time_limit: float,
    rng_seed: int | None,
    stochastic_tiebreak: bool,
):
    rng = random.Random(rng_seed) if rng_seed is not None else None
    tie_rng = rng if stochastic_tiebreak else None

    if config_name == "Baseline (Alpha-Beta)":
        return BaselineAgent(player, depth=depth, time_limit=time_limit, rng=tie_rng)

    if config_name == "No Pruning (Plain Minimax)":
        return NoPruningAgent(player, depth=depth, time_limit=time_limit, rng=tie_rng)

    raise ValueError(f"Unsupported configuration: {config_name}")

def _summarize_depth_config(
    depth: int,
    configuration: str,
    records: list[GameRecord],
) -> DepthConfigSummary:
    wins = sum(1 for r in records if r.config_won)
    losses = sum(1 for r in records if not r.config_won and not r.draw)
    draws = sum(1 for r in records if r.draw)
    
    node_values = [r.config_avg_nodes_per_move for r in records]
    time_values = [r.config_avg_time_per_move_ms for r in records]

    return DepthConfigSummary(
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

def run_experiment(
    games_per_side: int = NUM_GAMES,
    depths: list[int] | None = None,
    time_limit: float = TIME_LIMIT,
    stochastic_tiebreak: bool = True,
    opening_random_plies: int = 0,
    seed: int | None = None,
    verbose: bool = False,
) -> tuple[list[GameRecord], ExperimentSummary]:
    if depths is None:
        depths = DEPTHS

    depths = sorted(set(depths))
    configs = CONFIGS

    print("=" * 72)
    print("Experiment 4: Baseline vs No-Pruning")
    print(f"Depths={depths} | {games_per_side * 2} games per config-depth")
    print("Opponent: Random")
    print(f"Stochastic tie-break: {'ON' if stochastic_tiebreak else 'OFF'}")
    print(f"Opening random plies: {opening_random_plies}")
    if seed is not None:
        print(f"Seed={seed}")
    print("=" * 72)

    all_records: list[GameRecord] = []
    summaries: list[DepthConfigSummary] = []
    game_id = 0
    run_rng = random.Random(seed) if seed is not None else random.Random()

    for depth in depths:
        print(f"\n{'='*60}")
        print(f"Depth {depth}")
        print(f"{'='*60}")
        
        for configuration in configs:
            print(f"\n  {configuration} vs Random")
            bucket: list[GameRecord] = []

            for config_side in [RED, BLACK]:
                random_side = BLACK if config_side == RED else RED

                for game_num in range(games_per_side):
                    game_seed = run_rng.randrange(2**32)
                    agent_config = _build_agent(
                        configuration,
                        config_side,
                        depth,
                        time_limit,
                        rng_seed=game_seed ^ 0xA5A5A5A5,
                        stochastic_tiebreak=stochastic_tiebreak,
                    )
                    agent_random = RandomAgent(random_side, rng=random.Random(game_seed ^ 0x5A5A5A5A))
                    opening_rng = random.Random(game_seed ^ 0xC3C3C3C3)

                    print(f"    Playing game {game_num + 1}/{games_per_side} as {config_side}...", end=" ", flush=True)

                    if config_side == RED:
                        result = play_game(
                            agent_config,
                            agent_random,
                            tracked_player=RED,
                            opening_random_plies=opening_random_plies,
                            opening_rng=opening_rng,
                            verbose=verbose,
                        )
                    else:
                        result = play_game(
                            agent_random,
                            agent_config,
                            tracked_player=BLACK,
                            opening_random_plies=opening_random_plies,
                            opening_rng=opening_rng,
                            verbose=verbose,
                        )

                    record = GameRecord(
                        game_id=game_id,
                        depth=depth,
                        configuration=configuration,
                        config_side=config_side,
                        winner=result.winner,
                        config_won=(result.winner == config_side),
                        draw=result.draw,
                        draw_reason=result.draw_reason,
                        move_limit_hit=result.move_limit_hit,
                        total_moves=result.total_moves,
                        config_avg_nodes_per_move=(
                            result.avg_nodes_per_move_red
                            if config_side == RED
                            else result.avg_nodes_per_move_black
                        ),
                        config_avg_time_per_move_ms=(
                            result.avg_time_per_move_red_s * 1000.0
                            if config_side == RED
                            else result.avg_time_per_move_black_s * 1000.0
                        ),
                        random_avg_nodes_per_move=(
                            result.avg_nodes_per_move_black
                            if config_side == RED
                            else result.avg_nodes_per_move_red
                        ),
                        random_avg_time_per_move_ms=(
                            result.avg_time_per_move_black_s * 1000.0
                            if config_side == RED
                            else result.avg_time_per_move_red_s * 1000.0
                        ),
                    )

                    all_records.append(record)
                    bucket.append(record)
                    game_id += 1

                    status = "WIN" if record.config_won else ("DRAW" if record.draw else "LOSS")
                    print(f"{status} | nodes/move={record.config_avg_nodes_per_move:,.1f} | ms/move={record.config_avg_time_per_move_ms:.2f}")

            summary = _summarize_depth_config(depth, configuration, bucket)
            summaries.append(summary)
            
            print(f"\n  Summary: W={summary.wins} L={summary.losses} D={summary.draws} ({summary.win_rate_pct:.1f}%)")
            print(f"  Avg Nodes/Move: {summary.avg_nodes_per_move_mean:,.1f} ± {summary.avg_nodes_per_move_std:,.1f}")
            print(f"  Avg Time/Move:  {summary.avg_time_per_move_ms_mean:.2f} ± {summary.avg_time_per_move_ms_std:.2f} ms")

    # Calculate aggregate pruning efficiency
    baseline_records = [r for r in all_records if r.configuration == "Baseline (Alpha-Beta)"]
    nopruning_records = [r for r in all_records if r.configuration == "No Pruning (Plain Minimax)"]
    
    baseline_nodes = [r.config_avg_nodes_per_move for r in baseline_records]
    nopruning_nodes = [r.config_avg_nodes_per_move for r in nopruning_records]
    
    baseline_avg = _mean(baseline_nodes)
    nopruning_avg = _mean(nopruning_nodes)
    pruning_eff = ((nopruning_avg - baseline_avg) / nopruning_avg * 100.0) if nopruning_avg > 0 else 0.0

    print(f"\n{'='*72}")
    print("AGGREGATE RESULTS")
    print(f"{'='*72}")
    print(f"Baseline (Alpha-Beta) Avg Nodes:    {baseline_avg:,.1f}")
    print(f"NoPruning (Plain Minimax) Avg Nodes: {nopruning_avg:,.1f}")
    print(f"Pruning Efficiency:                  {pruning_eff:.1f}% nodes saved")
    print(f"{'='*72}")

    experiment_summary = ExperimentSummary(
        depths=depths,
        configurations=configs,
        total_games=len(all_records),
        games_per_side=games_per_side,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        depth_config_summaries=summaries,
        baseline_avg_nodes=baseline_avg,
        nopruning_avg_nodes=nopruning_avg,
        pruning_efficiency_pct=pruning_eff,
    )

    return all_records, experiment_summary

def save_results(
    records: list[GameRecord],
    summary: ExperimentSummary,
    results_dir: str = RESULTS_DIR,
) -> None:
    base_dir = os.path.join(results_dir, "experiment4_baseline_vs_nopruning")
    os.makedirs(base_dir, exist_ok=True)

    depth_span = f"{min(summary.depths)}to{max(summary.depths)}"
    run_folder = f"d{depth_span}_n{summary.games_per_side * 2}"
    run_dir = os.path.join(base_dir, run_folder)
    os.makedirs(run_dir, exist_ok=True)

    run_games_path = os.path.join(run_dir, "experiment4_baseline_vs_nopruning.json")
    run_summary_path = os.path.join(run_dir, "experiment4_summary.json")

    with open(run_games_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, indent=2)

    with open(run_summary_path, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)

    print(f"\nResults saved:")
    print(f"  {run_games_path}")
    print(f"  {run_summary_path}")

def _save_graphs(
    summary: ExperimentSummary,
    results_dir: str = RESULTS_DIR,
) -> list[str]:
    base_dir = os.path.join(results_dir, "experiment4_baseline_vs_nopruning")
    depth_span = f"{min(summary.depths)}to{max(summary.depths)}"
    run_folder = f"d{depth_span}_n{summary.games_per_side * 2}"
    images_dir = os.path.join(base_dir, run_folder, "images")
    os.makedirs(images_dir, exist_ok=True)

    output_paths: list[str] = []

    # Organize by config
    by_config: dict[str, list[DepthConfigSummary]] = {}
    for item in summary.depth_config_summaries:
        by_config.setdefault(item.configuration, []).append(item)

    for config in by_config:
        by_config[config] = sorted(by_config[config], key=lambda x: x.depth)

    # 1. Nodes comparison by depth
    nodes_path = os.path.join(images_dir, "baseline_vs_nopruning_nodes_comparison.png")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for config, points in by_config.items():
        x_values = [p.depth for p in points]
        means = [p.avg_nodes_per_move_mean for p in points]
        stds = [p.avg_nodes_per_move_std for p in points]
        
        ax.errorbar(
            x_values,
            means,
            yerr=stds,
            marker="o",
            capsize=5,
            label=config,
            color=CONFIG_COLORS.get(config, "#0072B2"),
            linewidth=2.5,
            markersize=8,
        )
    
    ax.set_title("Baseline vs No-Pruning: Nodes Expanded", fontsize=14, fontweight='bold')
    ax.set_xlabel("Search Depth", fontsize=12)
    ax.set_ylabel("Avg Nodes per Move", fontsize=12)
    ax.set_xticks(summary.depths)
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(frameon=True, fontsize=11)
    fig.tight_layout()
    fig.savefig(nodes_path, dpi=200)
    plt.close(fig)
    output_paths.append(nodes_path)

    # 2. Time comparison by depth
    time_path = os.path.join(images_dir, "baseline_vs_nopruning_time_comparison.png")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for config, points in by_config.items():
        x_values = [p.depth for p in points]
        means = [p.avg_time_per_move_ms_mean for p in points]
        stds = [p.avg_time_per_move_ms_std for p in points]
        
        ax.errorbar(
            x_values,
            means,
            yerr=stds,
            marker="o",
            capsize=5,
            label=config,
            color=CONFIG_COLORS.get(config, "#0072B2"),
            linewidth=2.5,
            markersize=8,
        )
    
    ax.set_title("Baseline vs No-Pruning: Computation Time", fontsize=14, fontweight='bold')
    ax.set_xlabel("Search Depth", fontsize=12)
    ax.set_ylabel("Avg Time per Move (ms)", fontsize=12)
    ax.set_xticks(summary.depths)
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(frameon=True, fontsize=11)
    fig.tight_layout()
    fig.savefig(time_path, dpi=200)
    plt.close(fig)
    output_paths.append(time_path)

    # 3. Pruning efficiency bar chart
    eff_path = os.path.join(images_dir, "pruning_efficiency_by_depth.png")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate pruning efficiency per depth
    efficiencies = []
    for depth in summary.depths:
        baseline = next((p for p in by_config.get("Baseline (Alpha-Beta)", []) if p.depth == depth), None)
        nopruning = next((p for p in by_config.get("No Pruning (Plain Minimax)", []) if p.depth == depth), None)
        
        if baseline and nopruning and nopruning.avg_nodes_per_move_mean > 0:
            eff = ((nopruning.avg_nodes_per_move_mean - baseline.avg_nodes_per_move_mean) 
                   / nopruning.avg_nodes_per_move_mean * 100.0)
            efficiencies.append(eff)
        else:
            efficiencies.append(0.0)
    
    bars = ax.bar(summary.depths, efficiencies, color="#2E8B57", alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=10, fontweight='bold')
    
    ax.set_title("Pruning Efficiency by Depth", fontsize=14, fontweight='bold')
    ax.set_xlabel("Search Depth", fontsize=12)
    ax.set_ylabel("Nodes Saved (%)", fontsize=12)
    ax.set_xticks(summary.depths)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=summary.pruning_efficiency_pct, color='red', linestyle='--', 
               linewidth=2, label=f'Overall Avg: {summary.pruning_efficiency_pct:.1f}%')
    ax.legend(frameon=True, fontsize=11)
    fig.tight_layout()
    fig.savefig(eff_path, dpi=200)
    plt.close(fig)
    output_paths.append(eff_path)

    return output_paths

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 4: Baseline vs No-Pruning")
    parser.add_argument("--games", type=int, default=NUM_GAMES, help="Games per side per depth/config")
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=DEPTHS,
        help="Depth values to test",
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
    parser.add_argument("--verbose", action="store_true", help="Print board each move")
    parser.add_argument("--no-save", action="store_true", help="Skip writing results to disk")
    parser.add_argument("--no-plot", action="store_true", help="Skip graph generation")
    args = parser.parse_args()

    records, summary = run_experiment(
        games_per_side=max(1, args.games),
        depths=args.depths,
        time_limit=args.time_limit,
        stochastic_tiebreak=args.stochastic_tiebreak,
        opening_random_plies=max(0, args.opening_random_plies),
        seed=args.seed,
        verbose=args.verbose,
    )

    if not args.no_save:
        save_results(records, summary)

    if args.no_plot:
        sys.exit(0)

    graph_paths = _save_graphs(summary)
    print("\nSaved graphs:")
    for path in graph_paths:
        print(f"  {path}")