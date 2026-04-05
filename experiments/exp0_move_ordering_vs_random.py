"""
Experiment 0: Move Ordering Agent vs. Random
=============================================

PURPOSE
-------
Compare the move-ordering agent against a random opponent
while holding depth and time limit constant.

METRICS COLLECTED PER GAME
--------------------------
  - Winner / draw outcome
  - Total moves in game
  - Nodes expanded by both agents
  - Time spent by both agents
  - Relative reductions for the move-ordering agent vs random

AGGREGATE STATISTICS REPORTED
-----------------------------
  - Win/Loss/Draw rates for Move Ordering Agent (mean percentages)
  - Avg nodes/move and time/move for both agents
  - Avg node/time reduction percentages (move ordering vs random)

OUTPUT
------
    results/exp0/move_ordering_vs_random/d{depth}_n{games}/exp0_move_ordering_vs_random.json
    results/exp0/move_ordering_vs_random/d{depth}_n{games}/exp0_move_ordering_random_summary.json

  plus latest snapshots:
    results/exp0/move_ordering_vs_random/exp0_move_ordering_vs_random.json
    results/exp0/move_ordering_vs_random/exp0_move_ordering_random_summary.json
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Optional

from src.agents.base import RandomAgent
from src.agents.move_ordering import MoveOrderingAgent
from src.engine.board import BLACK, RED, Board
from src.engine.game_runner import play_game, GameResult


NUM_GAMES = 50
DEPTH = 4
TIME_LIMIT = 5.0
MAX_MOVES = 300
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def _pct(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


def _reduction(random_value: float, ordering_value: float) -> float:
    if random_value == 0:
        return 0.0
    return (1.0 - ordering_value / random_value) * 100.0


@dataclass
class GameRecord:
    game_id: int
    ordering_side: str
    winner: str
    ordering_won: bool
    draw: bool
    draw_reason: str
    move_limit_hit: bool
    total_moves: int
    ordering_pieces_final: int
    random_pieces_final: int
    ordering_total_nodes: int
    random_total_nodes: int
    ordering_avg_nodes_per_move: float
    random_avg_nodes_per_move: float
    ordering_total_time_s: float
    random_total_time_s: float
    ordering_avg_time_per_move_ms: float
    random_avg_time_per_move_ms: float
    nodes_reduction_pct: float
    time_reduction_pct: float
    # Plot compatibility fields (plot_performance.py)
    avg_nodes_per_move: float
    avg_time_per_move_ms: float


@dataclass
class AggregateSummary:
    total_games: int
    wins: int
    losses: int
    draws: int
    win_rate_pct: float
    loss_rate_pct: float
    draw_rate_pct: float
    avg_nodes_per_move_mean: float
    avg_nodes_per_move_std: float
    avg_time_per_move_ms_mean: float
    avg_time_per_move_ms_std: float
    avg_nodes_per_move_random_mean: float
    avg_nodes_per_move_random_std: float
    avg_time_per_move_ms_random_mean: float
    avg_time_per_move_ms_random_std: float
    nodes_reduction_pct_mean: float
    nodes_reduction_pct_std: float
    time_reduction_pct_mean: float
    time_reduction_pct_std: float
    game_length_mean: float
    game_length_std: float
    baseline_depth: int
    num_games: int
    timestamp: str


def run_experiment(
    num_games: int = NUM_GAMES,
    depth: int = DEPTH,
    time_limit: float = TIME_LIMIT,
    verbose: bool = False,
) -> tuple[list[GameRecord], AggregateSummary]:
    """
    Play num_games games: half with Move Ordering as RED, half as BLACK.
    Opponent is always Random.
    """
    records: list[GameRecord] = []
    game_id = 0
    half = num_games // 2

    print(f"\n{'=' * 60}")
    print(f"Experiment 0: Move Ordering vs Random (depth={depth})")
    print(f"{'=' * 60}")
    print(f"Games with Move Ordering as RED  : {half}")

    # -- Move Ordering plays RED (half the games) --
    for _ in range(half):
        ordering = MoveOrderingAgent(RED, depth=depth, time_limit=time_limit)
        random_a = RandomAgent(BLACK)

        result = play_game(ordering, random_a, tracked_player=RED, verbose=verbose)

        ordering_avg_nodes = result.avg_nodes_per_move
        random_avg_nodes = 0.0  # Random doesn't expand nodes

        ordering_avg_time_ms = result.avg_time_per_move_s * 1000.0
        random_avg_time_ms = 0.0  # Random doesn't track meaningful time

        rec = GameRecord(
            game_id=game_id,
            ordering_side=RED,
            winner=result.winner,
            ordering_won=(result.winner == RED),
            draw=result.draw,
            draw_reason=result.draw_reason,
            move_limit_hit=result.move_limit_hit,
            total_moves=result.total_moves,
            ordering_pieces_final=result.baseline_pieces_final,
            random_pieces_final=result.opponent_pieces_final,
            ordering_total_nodes=result.total_nodes,
            random_total_nodes=0,
            ordering_avg_nodes_per_move=ordering_avg_nodes,
            random_avg_nodes_per_move=random_avg_nodes,
            ordering_total_time_s=result.total_time_s,
            random_total_time_s=0.0,
            ordering_avg_time_per_move_ms=ordering_avg_time_ms,
            random_avg_time_per_move_ms=random_avg_time_ms,
            nodes_reduction_pct=0.0,  # N/A vs random
            time_reduction_pct=0.0,   # N/A vs random
            avg_nodes_per_move=ordering_avg_nodes,
            avg_time_per_move_ms=ordering_avg_time_ms,
        )
        records.append(rec)
        game_id += 1

        status = "WIN" if rec.ordering_won else ("DRAW" if rec.draw else "LOSS")
        print(
            f"  Game {game_id:>3}  [ORD=RED]  {status:4}  "
            f"moves={rec.total_moves:>3}  "
            f"ord nodes/move={rec.ordering_avg_nodes_per_move:>8.1f}"
        )

    print(f"\nGames with Move Ordering as BLACK: {num_games - half}")
    for _ in range(num_games - half):
        random_a = RandomAgent(RED)
        ordering = MoveOrderingAgent(BLACK, depth=depth, time_limit=time_limit)

        result = play_game(random_a, ordering, tracked_player=BLACK, verbose=verbose)

        ordering_avg_nodes = result.avg_nodes_per_move
        random_avg_nodes = 0.0

        ordering_avg_time_ms = result.avg_time_per_move_s * 1000.0
        random_avg_time_ms = 0.0

        rec = GameRecord(
            game_id=game_id,
            ordering_side=BLACK,
            winner=result.winner,
            ordering_won=(result.winner == BLACK),
            draw=result.draw,
            draw_reason=result.draw_reason,
            move_limit_hit=result.move_limit_hit,
            total_moves=result.total_moves,
            ordering_pieces_final=result.baseline_pieces_final,
            random_pieces_final=result.opponent_pieces_final,
            ordering_total_nodes=result.total_nodes,
            random_total_nodes=0,
            ordering_avg_nodes_per_move=ordering_avg_nodes,
            random_avg_nodes_per_move=random_avg_nodes,
            ordering_total_time_s=result.total_time_s,
            random_total_time_s=0.0,
            ordering_avg_time_per_move_ms=ordering_avg_time_ms,
            random_avg_time_per_move_ms=random_avg_time_ms,
            nodes_reduction_pct=0.0,
            time_reduction_pct=0.0,
            avg_nodes_per_move=ordering_avg_nodes,
            avg_time_per_move_ms=ordering_avg_time_ms,
        )
        records.append(rec)
        game_id += 1

        status = "WIN" if rec.ordering_won else ("DRAW" if rec.draw else "LOSS")
        print(
            f"  Game {game_id:>3}  [ORD=BLK]  {status:4}  "
            f"moves={rec.total_moves:>3}  "
            f"ord nodes/move={rec.ordering_avg_nodes_per_move:>8.1f}"
        )

    summary = _build_summary(records, depth=depth, num_games=num_games)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total games           : {summary.total_games}")
    print(f"  Move Ordering wins    : {summary.wins}  ({summary.win_rate_pct:.1f}%)")
    print(f"  Move Ordering losses  : {summary.losses}  ({summary.loss_rate_pct:.1f}%)")
    print(f"  Draws                 : {summary.draws}  ({summary.draw_rate_pct:.1f}%)")
    print(
        "  Ordering nodes/move   : "
        f"{summary.avg_nodes_per_move_mean:.1f} ± {summary.avg_nodes_per_move_std:.1f}"
    )
    print(
        "  Ordering time/move (ms): "
        f"{summary.avg_time_per_move_ms_mean:.2f} ± {summary.avg_time_per_move_ms_std:.2f}"
    )

    return records, summary


def _build_summary(records: list[GameRecord], depth: int, num_games: int) -> AggregateSummary:
    wins = sum(1 for r in records if r.ordering_won)
    losses = sum(1 for r in records if not r.ordering_won and not r.draw)
    draws = sum(1 for r in records if r.draw)

    ord_nodes = [r.ordering_avg_nodes_per_move for r in records]
    ord_time = [r.ordering_avg_time_per_move_ms for r in records]
    game_lengths = [float(r.total_moves) for r in records]

    return AggregateSummary(
        total_games=len(records),
        wins=wins,
        losses=losses,
        draws=draws,
        win_rate_pct=_pct(wins, len(records)),
        loss_rate_pct=_pct(losses, len(records)),
        draw_rate_pct=_pct(draws, len(records)),
        avg_nodes_per_move_mean=_mean(ord_nodes),
        avg_nodes_per_move_std=_std(ord_nodes),
        avg_time_per_move_ms_mean=_mean(ord_time),
        avg_time_per_move_ms_std=_std(ord_time),
        avg_nodes_per_move_random_mean=0.0,
        avg_nodes_per_move_random_std=0.0,
        avg_time_per_move_ms_random_mean=0.0,
        avg_time_per_move_ms_random_std=0.0,
        nodes_reduction_pct_mean=0.0,
        nodes_reduction_pct_std=0.0,
        time_reduction_pct_mean=0.0,
        time_reduction_pct_std=0.0,
        game_length_mean=_mean(game_lengths),
        game_length_std=_std(game_lengths),
        baseline_depth=depth,
        num_games=num_games,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


def save_results(
    records: list[GameRecord],
    summary: AggregateSummary,
    results_dir: str = RESULTS_DIR,
) -> None:
    base_dir = os.path.join(results_dir, "exp0", "move_ordering_vs_random")
    os.makedirs(base_dir, exist_ok=True)

    run_folder = f"d{summary.baseline_depth}_n{summary.num_games}"
    run_dir = os.path.join(base_dir, run_folder)
    os.makedirs(run_dir, exist_ok=True)

    run_games_path = os.path.join(run_dir, "exp0_move_ordering_vs_random.json")
    run_summary_path = os.path.join(run_dir, "exp0_move_ordering_random_summary.json")

    latest_games_path = os.path.join(base_dir, "exp0_move_ordering_vs_random.json")
    latest_summary_path = os.path.join(base_dir, "exp0_move_ordering_random_summary.json")

    with open(run_games_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, indent=2)

    with open(run_summary_path, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)

    with open(latest_games_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, indent=2)

    with open(latest_summary_path, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)

    print("\nResults saved:")
    print(f"  {run_games_path}")
    print(f"  {run_summary_path}")
    print(f"  {latest_games_path}")
    print(f"  {latest_summary_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 0: Move Ordering vs Random")
    parser.add_argument("--games", type=int, default=NUM_GAMES, help="Total games to play")
    parser.add_argument("--depth", type=int, default=DEPTH, help="Search depth for Move Ordering")
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT, help="Time limit per move (s)")
    parser.add_argument("--verbose", action="store_true", help="Print board each move")
    parser.add_argument("--no-save", action="store_true", help="Skip writing results to disk")
    args = parser.parse_args()

    records, summary = run_experiment(
        num_games=args.games,
        depth=args.depth,
        time_limit=args.time_limit,
        verbose=args.verbose,
    )

    if not args.no_save:
        save_results(records, summary)
