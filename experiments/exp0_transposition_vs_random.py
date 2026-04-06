"""
Experiment 0: Transposition Agent vs. Random
=============================================

PURPOSE
-------
Compare the transposition-table agent against a random opponent
while holding depth and time limit constant.

METRICS COLLECTED PER GAME
--------------------------
  - Winner / draw outcome
  - Total moves in game
  - Nodes expanded by the transposition agent
  - Time spent by the transposition agent
  - Transposition-table hit count

AGGREGATE STATISTICS REPORTED
-----------------------------
  - Win/Loss/Draw rates for the transposition agent (mean percentages)
  - Avg nodes/move and time/move
  - Avg TT hits per move and TT hit rate

OUTPUT
------
    results/exp0/transposition_vs_random/d{depth}_n{games}/exp0_transposition_vs_random.json
    results/exp0/transposition_vs_random/d{depth}_n{games}/exp0_transposition_summary.json

  plus latest snapshots:
    results/exp0/transposition_vs_random/exp0_transposition_vs_random.json
    results/exp0/transposition_vs_random/exp0_transposition_summary.json
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass

from src.agents.base import RandomAgent
from src.agents.transposition import TranspositionAgent
from src.engine.board import BLACK, RED, Board


NUM_GAMES = 50
DEPTH = 4
TIME_LIMIT = 5.0
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_value = _mean(values)
    return math.sqrt(sum((value - mean_value) ** 2 for value in values) / (len(values) - 1))


def _pct(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


@dataclass
class GameRecord:
    game_id: int
    transposition_side: str
    winner: str
    transposition_won: bool
    draw: bool
    draw_reason: str
    move_limit_hit: bool
    total_moves: int
    transposition_pieces_final: int
    random_pieces_final: int
    transposition_total_nodes: int
    transposition_avg_nodes_per_move: float
    transposition_total_time_s: float
    transposition_avg_time_per_move_ms: float
    transposition_tt_hits: int
    transposition_tt_hit_rate_pct: float
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
    avg_tt_hits_mean: float
    avg_tt_hits_std: float
    avg_tt_hit_rate_pct_mean: float
    avg_tt_hit_rate_pct_std: float
    game_length_mean: float
    game_length_std: float
    baseline_depth: int
    transposition_depth: int
    num_games: int
    timestamp: str


def _play_game(
    transposition_side: str,
    depth: int,
    time_limit: float,
    verbose: bool,
) -> GameRecord:
    board = Board()

    if transposition_side == RED:
        transposition_agent = TranspositionAgent(RED, depth=depth, time_limit=time_limit)
        random_agent = RandomAgent(BLACK)
    else:
        transposition_agent = TranspositionAgent(BLACK, depth=depth, time_limit=time_limit)
        random_agent = RandomAgent(RED)

    agents = {RED: random_agent, BLACK: random_agent}
    agents[transposition_side] = transposition_agent

    transposition_nodes_per_move: list[int] = []
    transposition_time_per_move_s: list[float] = []
    total_transposition_tt_hits = 0
    total_moves = 0

    state_history: dict[str, int] = {}
    no_progress_count = 0
    draw_reason = ""

    while not board.is_terminal() and total_moves < 300:
        current = board.current_player
        agent = agents[current]

        if hasattr(agent, "nodes_expanded"):
            agent.nodes_expanded = 0

        t0 = time.perf_counter()
        move = agent.choose_move(board)
        elapsed = time.perf_counter() - t0

        if move is None:
            break

        if current == transposition_side:
            transposition_nodes_per_move.append(getattr(agent, "nodes_expanded", 0))
            transposition_time_per_move_s.append(elapsed)
            total_transposition_tt_hits += getattr(agent, "tt_hits", 0)

        board = board.apply_move(move)
        total_moves += 1

        if verbose:
            print(f"\nMove {total_moves} ({current}):")
            print(board)

        if len(move.captures) > 0:
            no_progress_count = 0
        else:
            no_progress_count += 1
            if no_progress_count >= 50:
                draw_reason = "no_progress"
                break

        board_state_key = str(board.grid) + "|" + board.current_player
        state_history[board_state_key] = state_history.get(board_state_key, 0) + 1
        if state_history[board_state_key] >= 3:
            draw_reason = "threefold_repetition"
            break

    move_limit_hit = total_moves >= 300
    terminal = board._terminal_result()

    if draw_reason:
        winner = "DRAW"
        draw = True
    elif terminal is None or move_limit_hit:
        winner = "DRAW"
        draw = True
        draw_reason = "move_limit" if move_limit_hit else "stalemate"
    elif terminal == "DRAW":
        winner = "DRAW"
        draw = True
        draw_reason = "terminal_draw"
    else:
        winner = terminal
        draw = False
        draw_reason = "terminal_win"

    total_nodes = sum(transposition_nodes_per_move)
    total_time_s = sum(transposition_time_per_move_s)
    avg_nodes_per_move = total_nodes / len(transposition_nodes_per_move) if transposition_nodes_per_move else 0.0
    avg_time_per_move_ms = (
        (total_time_s / len(transposition_time_per_move_s)) * 1000.0
        if transposition_time_per_move_s
        else 0.0
    )
    tt_hit_rate_pct = _pct(total_transposition_tt_hits, total_moves) if total_moves else 0.0

    opponent_side = BLACK if transposition_side == RED else RED

    return GameRecord(
        game_id=-1,
        transposition_side=transposition_side,
        winner=winner,
        transposition_won=(winner == transposition_side),
        draw=draw,
        draw_reason=draw_reason,
        move_limit_hit=move_limit_hit,
        total_moves=total_moves,
        transposition_pieces_final=board.count_pieces(transposition_side),
        random_pieces_final=board.count_pieces(opponent_side),
        transposition_total_nodes=total_nodes,
        transposition_avg_nodes_per_move=avg_nodes_per_move,
        transposition_total_time_s=total_time_s,
        transposition_avg_time_per_move_ms=avg_time_per_move_ms,
        transposition_tt_hits=total_transposition_tt_hits,
        transposition_tt_hit_rate_pct=tt_hit_rate_pct,
        avg_nodes_per_move=avg_nodes_per_move,
        avg_time_per_move_ms=avg_time_per_move_ms,
    )


def _build_summary(records: list[GameRecord], depth: int, num_games: int) -> AggregateSummary:
    wins = sum(1 for record in records if record.transposition_won)
    losses = sum(1 for record in records if not record.transposition_won and not record.draw)
    draws = sum(1 for record in records if record.draw)

    nodes_per_move = [record.transposition_avg_nodes_per_move for record in records]
    time_per_move = [record.transposition_avg_time_per_move_ms for record in records]
    tt_hits = [float(record.transposition_tt_hits) for record in records]
    tt_hit_rates = [record.transposition_tt_hit_rate_pct for record in records]
    game_lengths = [float(record.total_moves) for record in records]

    return AggregateSummary(
        total_games=len(records),
        wins=wins,
        losses=losses,
        draws=draws,
        win_rate_pct=_pct(wins, len(records)),
        loss_rate_pct=_pct(losses, len(records)),
        draw_rate_pct=_pct(draws, len(records)),
        avg_nodes_per_move_mean=_mean(nodes_per_move),
        avg_nodes_per_move_std=_std(nodes_per_move),
        avg_time_per_move_ms_mean=_mean(time_per_move),
        avg_time_per_move_ms_std=_std(time_per_move),
        avg_tt_hits_mean=_mean(tt_hits),
        avg_tt_hits_std=_std(tt_hits),
        avg_tt_hit_rate_pct_mean=_mean(tt_hit_rates),
        avg_tt_hit_rate_pct_std=_std(tt_hit_rates),
        game_length_mean=_mean(game_lengths),
        game_length_std=_std(game_lengths),
        baseline_depth=depth,
        transposition_depth=depth,
        num_games=num_games,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


def run_experiment(
    num_games: int = NUM_GAMES,
    depth: int = DEPTH,
    time_limit: float = TIME_LIMIT,
    verbose: bool = False,
) -> tuple[list[GameRecord], AggregateSummary]:
    records: list[GameRecord] = []
    game_id = 0
    half = num_games // 2

    print(f"\n{'=' * 60}")
    print(f"Experiment 0: Transposition Agent (depth={depth}) vs Random Agent")
    print(f"{'=' * 60}")
    print(f"Games as RED  : {half}")

    for _ in range(half):
        rec = _play_game(RED, depth=depth, time_limit=time_limit, verbose=verbose)
        rec.game_id = game_id
        records.append(rec)
        game_id += 1

        status = "WIN" if rec.transposition_won else ("DRAW" if rec.draw else "LOSS")
        print(
            f"  Game {game_id:>3}  [RED]  {status:4}  "
            f"moves={rec.total_moves:>3}  "
            f"nodes/move={rec.transposition_avg_nodes_per_move:>8.1f}  "
            f"ms/move={rec.transposition_avg_time_per_move_ms:>7.2f}"
        )

    print(f"\nGames as BLACK: {num_games - half}")
    for _ in range(num_games - half):
        rec = _play_game(BLACK, depth=depth, time_limit=time_limit, verbose=verbose)
        rec.game_id = game_id
        records.append(rec)
        game_id += 1

        status = "WIN" if rec.transposition_won else ("DRAW" if rec.draw else "LOSS")
        print(
            f"  Game {game_id:>3}  [BLK]  {status:4}  "
            f"moves={rec.total_moves:>3}  "
            f"nodes/move={rec.transposition_avg_nodes_per_move:>8.1f}  "
            f"ms/move={rec.transposition_avg_time_per_move_ms:>7.2f}"
        )

    summary = _build_summary(records, depth=depth, num_games=num_games)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total games               : {summary.total_games}")
    print(f"  Transposition wins        : {summary.wins}  ({summary.win_rate_pct:.1f}%)")
    print(f"  Transposition losses      : {summary.losses}  ({summary.loss_rate_pct:.1f}%)")
    print(f"  Draws                     : {summary.draws}  ({summary.draw_rate_pct:.1f}%)")
    print(
        "  Nodes/move                : "
        f"{summary.avg_nodes_per_move_mean:.1f} ± {summary.avg_nodes_per_move_std:.1f}"
    )
    print(
        "  Time/move (ms)            : "
        f"{summary.avg_time_per_move_ms_mean:.2f} ± {summary.avg_time_per_move_ms_std:.2f}"
    )
    print(
        "  TT hits / game            : "
        f"{summary.avg_tt_hits_mean:.1f} ± {summary.avg_tt_hits_std:.1f}"
    )
    print(
        "  TT hit rate (%)           : "
        f"{summary.avg_tt_hit_rate_pct_mean:.1f} ± {summary.avg_tt_hit_rate_pct_std:.1f}"
    )

    return records, summary


def save_results(
    records: list[GameRecord],
    summary: AggregateSummary,
    results_dir: str = RESULTS_DIR,
) -> None:
    base_dir = os.path.join(results_dir, "exp0", "transposition_vs_random")
    os.makedirs(base_dir, exist_ok=True)

    run_folder = f"d{summary.transposition_depth}_n{summary.num_games}"
    run_dir = os.path.join(base_dir, run_folder)
    os.makedirs(run_dir, exist_ok=True)

    run_games_path = os.path.join(run_dir, "exp0_transposition_vs_random.json")
    run_summary_path = os.path.join(run_dir, "exp0_transposition_summary.json")

    latest_games_path = os.path.join(base_dir, "exp0_transposition_vs_random.json")
    latest_summary_path = os.path.join(base_dir, "exp0_transposition_summary.json")

    with open(run_games_path, "w", encoding="utf-8") as f:
        json.dump([asdict(record) for record in records], f, indent=2)

    with open(run_summary_path, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)

    with open(latest_games_path, "w", encoding="utf-8") as f:
        json.dump([asdict(record) for record in records], f, indent=2)

    with open(latest_summary_path, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)

    print("\nResults saved:")
    print(f"  {run_games_path}")
    print(f"  {run_summary_path}")
    print(f"  {latest_games_path}")
    print(f"  {latest_summary_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 0: Transposition vs Random")
    parser.add_argument("--games", type=int, default=NUM_GAMES, help="Total games to play")
    parser.add_argument("--depth", type=int, default=DEPTH, help="Search depth for Transposition")
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
