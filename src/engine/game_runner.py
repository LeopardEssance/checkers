"""
Game runner utility.

Plays a complete game between two agents and returns a GameResult with
all per-move statistics needed for downstream analysis and charting.
"""

from __future__ import annotations

import time
import random
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
    nodes_per_move_red: list[int] = field(default_factory=list)
    time_per_move_red_s: list[float] = field(default_factory=list)
    nodes_per_move_black: list[int] = field(default_factory=list)
    time_per_move_black_s: list[float] = field(default_factory=list)
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

    @property
    def avg_nodes_per_move_red(self) -> float:
        return sum(self.nodes_per_move_red) / len(self.nodes_per_move_red) if self.nodes_per_move_red else 0.0

    @property
    def avg_time_per_move_red_s(self) -> float:
        return sum(self.time_per_move_red_s) / len(self.time_per_move_red_s) if self.time_per_move_red_s else 0.0

    @property
    def avg_nodes_per_move_black(self) -> float:
        return sum(self.nodes_per_move_black) / len(self.nodes_per_move_black) if self.nodes_per_move_black else 0.0

    @property
    def avg_time_per_move_black_s(self) -> float:
        return sum(self.time_per_move_black_s) / len(self.time_per_move_black_s) if self.time_per_move_black_s else 0.0

def play_game(
    agent_red,
    agent_black,
    tracked_player: str = RED,
    opening_random_plies: int = 0,
    opening_rng: Optional[random.Random] = None,
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
    opening_random_plies : number of opening plies selected randomly before agents decide
    opening_rng    : RNG for randomized opening plies
    max_moves      : draw declared if this many half-moves are reached
    verbose        : print board state each move

    Returns
    -------
    GameResult with all per-move statistics and draw_reason populated.
    """
    board  = Board()
    agents = {RED: agent_red, BLACK: agent_black}
    if opening_rng is None:
        opening_rng = random.Random()

    nodes_per_move:    list[int]   = []
    time_per_move_s:   list[float] = []
    nodes_per_move_red: list[int] = []
    time_per_move_red_s: list[float] = []
    nodes_per_move_black: list[int] = []
    time_per_move_black_s: list[float] = []
    total_moves = 0

    # Track board states for threefold repetition rule
    state_history: dict[str, int] = {}
    
    # Track consecutive moves with no capture for 50-move rule
    no_progress_count = 0

    draw_reason = ""

    while not board.is_terminal() and total_moves < max_moves:
        current = board.current_player
        agent   = agents[current]

        use_random_opening = total_moves < opening_random_plies

        # Reset node counter if the agent tracks it
        if not use_random_opening and hasattr(agent, "nodes_expanded"):
            agent.nodes_expanded = 0

        if use_random_opening:
            # During opening plies, pick a uniformly random move
            legal_moves = board.get_legal_moves(current)
            move = opening_rng.choice(legal_moves) if legal_moves else None
            elapsed = 0.0
        else:
            # Let agent choose (time and nodes will be measured)
            t0 = time.perf_counter()
            move = agent.choose_move(board)
            elapsed = time.perf_counter() - t0

        if move is None:
            # Agent has no moves — terminal will be caught on next iteration
            break

        nodes = 0 if use_random_opening else getattr(agent, "nodes_expanded", 0)

        # Record stats for both players (used by head-to-head plotting).
        if current == RED:
            nodes_per_move_red.append(nodes)
            time_per_move_red_s.append(elapsed)
        else:
            nodes_per_move_black.append(nodes)
            time_per_move_black_s.append(elapsed)

        # Preserve tracked-player stats for existing experiments.
        if current == tracked_player and not use_random_opening:
            nodes_per_move.append(nodes)
            time_per_move_s.append(elapsed)

        board = board.apply_move(move)
        total_moves += 1

        if verbose:
            print(f"\nMove {total_moves} ({current}):")
            print(board)

        # === No-progress rule (50-move rule): reset counter only on captures ===
        # In checkers, captures are the primary "progress" indicator.
        # After 50 consecutive non-capturing moves, the game is a draw.
        if len(move.captures) > 0:
            no_progress_count = 0
        else:
            # No capture made — increment counter
            no_progress_count += 1
            if no_progress_count >= 50:
                draw_reason = "no_progress"
                break

        # === Threefold repetition rule ===
        # If the same board state (grid + current_player) appears 3 times, game is a draw.
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
        nodes_per_move_red   = nodes_per_move_red,
        time_per_move_red_s  = time_per_move_red_s,
        nodes_per_move_black = nodes_per_move_black,
        time_per_move_black_s= time_per_move_black_s,
        baseline_pieces_final= board.count_pieces(tracked_player),
        opponent_pieces_final= board.count_pieces(tracked_opponent),
        draw                 = draw,
        draw_reason          = draw_reason,
        move_limit_hit       = move_limit_hit,
    )
