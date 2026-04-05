"""
Experiment 0: Baseline Sanity Check — BaselineAgent vs. RandomAgent

RUN WITH: python -m experiments.exp0_baseline_vs_random
TODO note to self: add another way to run this
====================================================================

PURPOSE
-------
This is a prerequisite validity check, not a comparison experiment.
Before any enhancement can be evaluated, the baseline agent must
demonstrate near-perfect dominance over a random opponent.

METRICS COLLECTED PER GAME
-----------------------------
  - Win / Loss / Draw outcome
  - Total moves in game
  - Nodes expanded per baseline move
  - Wall-clock time per baseline move in seconds
  - Remaining pieces for both sides at game end

AGGREGATE STATISTICS REPORTED
-------------------------------
  - Win rate, Loss rate, Draw rate  (mean ± std dev)
  - Avg nodes/move                  (mean ± std dev)
  - Avg time/move (ms)              (mean ± std dev)
  - Avg game length (moves)         (mean ± std dev)
  - Avg pieces remaining            (mean ± std dev)

OUTPUT
------
  results/exp0_baseline_vs_random.json   — full per-game data
  results/exp0_summary.json              — aggregate stats for report/charts
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

from src.agents.base        import BaselineAgent, RandomAgent
from src.engine.board       import RED, BLACK
from src.engine.game_runner import play_game, GameResult

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NUM_GAMES      = 50     # number of games per side-assignment
BASELINE_DEPTH = 4      
TIME_LIMIT     = 5.0    # max seconds per baseline move
RESULTS_DIR    = os.path.join(os.path.dirname(__file__), "..", "results")

# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0

def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))

def _pct(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0

# ---------------------------------------------------------------------------
# Per-game record (JSON-serializable)
# ---------------------------------------------------------------------------

@dataclass
class GameRecord:
    game_id:               int
    baseline_side:         str          # RED or BLACK
    winner:                str
    baseline_won:          bool
    draw:                  bool
    move_limit_hit:        bool
    total_moves:           int
    baseline_pieces_final: int
    opponent_pieces_final: int
    total_nodes:           int
    avg_nodes_per_move:    float
    total_time_s:          float
    avg_time_per_move_ms:  float        # stored as ms for readability
    nodes_per_move:        list[int]
    time_per_move_ms:      list[float]

def _record_from_result(
    game_id: int,
    baseline_side: str,
    result: GameResult,
) -> GameRecord:
    return GameRecord(
        game_id               = game_id,
        baseline_side         = baseline_side,
        winner                = result.winner,
        baseline_won          = (result.winner == baseline_side),
        draw                  = result.draw,
        move_limit_hit        = result.move_limit_hit,
        total_moves           = result.total_moves,
        baseline_pieces_final = result.baseline_pieces_final,
        opponent_pieces_final = result.opponent_pieces_final,
        total_nodes           = result.total_nodes,
        avg_nodes_per_move    = result.avg_nodes_per_move,
        total_time_s          = result.total_time_s,
        avg_time_per_move_ms  = result.avg_time_per_move_s * 1000,
        nodes_per_move        = result.nodes_per_move,
        time_per_move_ms      = [t * 1000 for t in result.time_per_move_s],
    )

# ---------------------------------------------------------------------------
# Aggregate summary
# ---------------------------------------------------------------------------

@dataclass
class AggregateSummary:
    """All aggregate stats with mean and std dev — ready for charts/tables."""
    total_games:  int
    # Outcomes
    wins:         int
    losses:       int
    draws:        int
    win_rate_pct: float
    loss_rate_pct: float
    draw_rate_pct: float
    # Nodes
    avg_nodes_per_move_mean: float
    avg_nodes_per_move_std:  float
    total_nodes_mean:        float
    total_nodes_std:         float
    # Time
    avg_time_per_move_ms_mean: float
    avg_time_per_move_ms_std:  float
    total_time_s_mean:         float
    total_time_s_std:          float
    # Game length
    game_length_mean: float
    game_length_std:  float
    # Piece counts at end
    baseline_pieces_final_mean: float
    baseline_pieces_final_std:  float
    opponent_pieces_final_mean: float
    opponent_pieces_final_std:  float
    # Configuration echo
    baseline_depth: int
    num_games:      int
    timestamp:      str

def _build_summary(records: list[GameRecord]) -> AggregateSummary:
    n      = len(records)
    wins   = sum(1 for r in records if r.baseline_won)
    losses = sum(1 for r in records if not r.baseline_won and not r.draw)
    draws  = sum(1 for r in records if r.draw)

    nodes_per_move_avgs  = [r.avg_nodes_per_move      for r in records]
    total_nodes_list     = [float(r.total_nodes)       for r in records]
    time_per_move_ms_avg = [r.avg_time_per_move_ms     for r in records]
    total_time_s_list    = [r.total_time_s             for r in records]
    game_lengths         = [float(r.total_moves)       for r in records]
    baseline_pieces      = [float(r.baseline_pieces_final) for r in records]
    opponent_pieces      = [float(r.opponent_pieces_final) for r in records]

    return AggregateSummary(
        total_games  = n,
        wins         = wins,
        losses       = losses,
        draws        = draws,
        win_rate_pct  = _pct(wins,   n),
        loss_rate_pct = _pct(losses, n),
        draw_rate_pct = _pct(draws,  n),

        avg_nodes_per_move_mean = _mean(nodes_per_move_avgs),
        avg_nodes_per_move_std  = _std(nodes_per_move_avgs),
        total_nodes_mean        = _mean(total_nodes_list),
        total_nodes_std         = _std(total_nodes_list),

        avg_time_per_move_ms_mean = _mean(time_per_move_ms_avg),
        avg_time_per_move_ms_std  = _std(time_per_move_ms_avg),
        total_time_s_mean         = _mean(total_time_s_list),
        total_time_s_std          = _std(total_time_s_list),

        game_length_mean = _mean(game_lengths),
        game_length_std  = _std(game_lengths),

        baseline_pieces_final_mean = _mean(baseline_pieces),
        baseline_pieces_final_std  = _std(baseline_pieces),
        opponent_pieces_final_mean = _mean(opponent_pieces),
        opponent_pieces_final_std  = _std(opponent_pieces),

        baseline_depth = BASELINE_DEPTH,
        num_games      = NUM_GAMES,
        timestamp      = time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_experiment(
    num_games: int      = NUM_GAMES,
    depth: int          = BASELINE_DEPTH,
    time_limit: float   = TIME_LIMIT,
    verbose: bool       = False,
) -> tuple[list[GameRecord], AggregateSummary]:
    """
    Play `num_games` games with baseline as RED, then `num_games` with
    baseline as BLACK (balanced side assignment). Returns all records
    and the aggregate summary.
    """
    records:    list[GameRecord] = []
    game_id = 0
    half    = num_games // 2

    # -- Baseline plays RED (half the games) --
    print(f"\n{'='*60}")
    print(f"Experiment 0: Baseline (depth={depth}) vs Random Agent")
    print(f"{'='*60}")
    print(f"Games as RED  : {half}")

    for i in range(half):
        baseline = BaselineAgent(RED,   depth=depth, time_limit=time_limit)
        random_a = RandomAgent(BLACK)
        result   = play_game(baseline, random_a, tracked_player=RED, verbose=verbose)
        rec      = _record_from_result(game_id, RED, result)
        records.append(rec)
        game_id += 1

        status = "WIN" if rec.baseline_won else ("DRAW" if rec.draw else "LOSS")
        print(f"  Game {game_id:>3}  [RED]  {status:4}  "
              f"moves={rec.total_moves:>3}  "
              f"nodes/move={rec.avg_nodes_per_move:>8.1f}  "
              f"ms/move={rec.avg_time_per_move_ms:>7.2f}")

    # -- Baseline plays BLACK (other half) --
    print(f"\nGames as BLACK: {num_games - half}")

    for i in range(num_games - half):
        random_a = RandomAgent(RED)
        baseline = BaselineAgent(BLACK, depth=depth, time_limit=time_limit)
        result   = play_game(random_a, baseline, tracked_player=BLACK, verbose=verbose)
        rec      = _record_from_result(game_id, BLACK, result)
        records.append(rec)
        game_id += 1

        status = "WIN" if rec.baseline_won else ("DRAW" if rec.draw else "LOSS")
        print(f"  Game {game_id:>3}  [BLK]  {status:4}  "
              f"moves={rec.total_moves:>3}  "
              f"nodes/move={rec.avg_nodes_per_move:>8.1f}  "
              f"ms/move={rec.avg_time_per_move_ms:>7.2f}")

    summary = _build_summary(records)

    # -- Print summary --
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Total games : {summary.total_games}")
    print(f"  Wins        : {summary.wins}  ({summary.win_rate_pct:.1f}%)")
    print(f"  Losses      : {summary.losses}  ({summary.loss_rate_pct:.1f}%)")
    print(f"  Draws       : {summary.draws}  ({summary.draw_rate_pct:.1f}%)")
    print(f"  Nodes/move  : {summary.avg_nodes_per_move_mean:.1f} ± {summary.avg_nodes_per_move_std:.1f}")
    print(f"  Time/move   : {summary.avg_time_per_move_ms_mean:.2f} ± {summary.avg_time_per_move_ms_std:.2f} ms")
    print(f"  Game length : {summary.game_length_mean:.1f} ± {summary.game_length_std:.1f} moves")
    print(f"  Sanity check: {'PASS ✓' if summary.win_rate_pct >= 90 else 'FAIL ✗ — win rate below 90%'}")

    return records, summary

# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_results(
    records: list[GameRecord],
    summary: AggregateSummary,
    results_dir: str = RESULTS_DIR,
) -> None:
    os.makedirs(results_dir, exist_ok=True)

    games_path   = os.path.join(results_dir, "exp0_baseline_vs_random.json")
    summary_path = os.path.join(results_dir, "exp0_summary.json")

    with open(games_path, "w") as f:
        json.dump([asdict(r) for r in records], f, indent=2)

    with open(summary_path, "w") as f:
        json.dump(asdict(summary), f, indent=2)

    print(f"\nResults saved:")
    print(f"  {games_path}")
    print(f"  {summary_path}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 0: Baseline vs Random")
    parser.add_argument("--games",      type=int,   default=NUM_GAMES,      help="Total games to play")
    parser.add_argument("--depth",      type=int,   default=BASELINE_DEPTH, help="Baseline search depth")
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT,     help="Time limit per move (s)")
    parser.add_argument("--verbose",    action="store_true",                 help="Print board each move")
    parser.add_argument("--no-save",    action="store_true",                 help="Skip writing results to disk")
    args = parser.parse_args()

    records, summary = run_experiment(
        num_games  = args.games,
        depth      = args.depth,
        time_limit = args.time_limit,
        verbose    = args.verbose,
    )

    if not args.no_save:
        save_results(records, summary)
