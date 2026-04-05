"""
Game runner utility.

Plays a complete game between two agents and returns a GameResult with
all per-move statistics needed for downstream analysis and charting.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from src.engine.board import Board, RED, BLACK
from src.agents.base  import BaselineAgent, RandomAgent

# Maximum moves before declaring a draw to prevent infinite loops
MAX_MOVES = 300

@dataclass
class GameResult:
    """
    All statistics captured from a single completed game.

    winner            : RED, BLACK, or 'DRAW'
    total_moves       : half-moves (plies) played before terminal
    nodes_per_move    : nodes expanded by the tracked agent each move
    time_per_move_s   : wall-clock seconds spent by tracked agent each move
    baseline_pieces_final  : pieces remaining for the baseline agent at end
    opponent_pieces_final  : pieces remaining for the opponent at end
    draw              : True when the game ended by draw rule
    draw_reason       : 'terminal_win', 'threefold_repetition', 'no_progress', or 'move_limit'
    move_limit_hit    : True when MAX_MOVES was reached
    """
    winner: str
    total_moves: int
    nodes_per_move: list[int]   = field(default_factory=list)
    time_per_move_s: list[float] = field(default_factory=list)
    baseline_pieces_final: int  = 0
    opponent_pieces_final: int  = 0
    draw: bool                  = False
    draw_reason: str            = ""
    move_limit_hit: bool        = False

    # ------------------------------------------------------------------
    # Derived convenience properties
    # ------------------------------------------------------------------

    @property
    def total_nodes(self) -> int:
        return sum(self.nodes_per_move)

    @property
    def avg_nodes_per_move(self) -> float:
        return self.total_nodes / len(self.nodes_per_move) if self.nodes_per_move else 0.0

    @property
    def total_time_s(self) -> float:
        return sum(self.time_per_move_s)

    @property
    def avg_time_per_move_s(self) -> float:
        return self.total_time_s / len(self.time_per_move_s) if self.time_per_move_s else 0.0

def play_game(
    agent_red,
    agent_black,
    tracked_player: str = RED,
    max_moves: int = MAX_MOVES,
    verbose: bool = False,
) -> GameResult:
    """
    Play one complete game between agent_red and agent_black.

    Implements standard draw rules:
    - Threefold repetition: same board state (+ player) appears 3 times
    - No-progress rule: 50 consecutive moves with no capture/promotion
    - Move limit: fallback hard cap at max_moves

    Parameters
    ----------
    agent_red      : agent object with .choose_move(board) and .player == RED
    agent_black    : agent object with .choose_move(board) and .player == BLACK
    tracked_player : which side's node/time stats to record (default RED)
    max_moves      : draw declared if this many half-moves are reached
    verbose        : print board state each move

    Returns
    -------
    GameResult with all per-move statistics and draw_reason populated.
    """
    board  = Board()
    agents = {RED: agent_red, BLACK: agent_black}

    nodes_per_move:    list[int]   = []
    time_per_move_s:   list[float] = []
    total_moves = 0

    # Track board states for threefold repetition rule
    state_history: dict[str, int] = {}
    
    # Track consecutive moves with no capture for 50-move rule
    no_progress_count = 0

    draw_reason = ""

    while not board.is_terminal() and total_moves < max_moves:
        current = board.current_player
        agent   = agents[current]

        # Reset node counter if the agent tracks it
        if hasattr(agent, "nodes_expanded"):
            agent.nodes_expanded = 0

        t0   = time.perf_counter()
        move = agent.choose_move(board)
        elapsed = time.perf_counter() - t0

        if move is None:
            # Agent has no moves — terminal will be caught on next iteration
            break

        # Record stats only for the tracked player
        if current == tracked_player:
            nodes = getattr(agent, "nodes_expanded", 0)
            nodes_per_move.append(nodes)
            time_per_move_s.append(elapsed)

        board = board.apply_move(move)
        total_moves += 1

        if verbose:
            print(f"\nMove {total_moves} ({current}):")
            print(board)

        # === No-progress rule (50-move rule): reset counter only on captures ===
        # In checkers, captures are the primary "progress" indicator
        if len(move.captures) > 0:
            no_progress_count = 0
        else:
            # No capture made — increment counter
            no_progress_count += 1
            if no_progress_count >= 50:
                draw_reason = "no_progress"
                break

        # === Threefold repetition rule ===
        board_state_key = str(board.grid) + "|" + board.current_player
        state_history[board_state_key] = state_history.get(board_state_key, 0) + 1
        if state_history[board_state_key] >= 3:
            draw_reason = "threefold_repetition"
            break

    # ------------------------------------------------------------------
    # Determine outcome
    # ------------------------------------------------------------------
    move_limit_hit = total_moves >= max_moves

    # If draw_reason was already set (threefold or no-progress), use it
    if draw_reason:
        winner = "DRAW"
        draw = True
    else:
        result = board._terminal_result()
        if result is None or move_limit_hit:
            winner = "DRAW"
            draw = True
            draw_reason = "move_limit" if move_limit_hit else "stalemate"
        elif result == "DRAW":
            winner = "DRAW"
            draw = True
            draw_reason = "terminal_draw"
        else:
            winner = result
            draw = False
            draw_reason = "terminal_win"

    tracked_opponent = BLACK if tracked_player == RED else RED

    return GameResult(
        winner               = winner,
        total_moves          = total_moves,
        nodes_per_move       = nodes_per_move,
        time_per_move_s      = time_per_move_s,
        baseline_pieces_final= board.count_pieces(tracked_player),
        opponent_pieces_final= board.count_pieces(tracked_opponent),
        draw                 = draw,
        draw_reason          = draw_reason,
        move_limit_hit       = move_limit_hit,
    )
