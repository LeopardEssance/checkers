"""
Baseline Agent: Minimax + Alpha-Beta Pruning

Implements:
  - Heuristic evaluation function h(s) = w1*M + w2*K + w3*L + w4*P + w5*C
  - Minimax search with alpha-beta pruning at fixed depth
  - RandomAgent opponent for testing

This is the un optimized reference baseline. All metrics from enhancements
are measured against this implementation.
"""

import time
import random
from typing import Optional

from src.engine.board import Board, Move, RED, BLACK, WIN_SCORE, LOSS_SCORE, DRAW_SCORE

# ---------------------------------------------------------------------------
# Heuristic weights
# ---------------------------------------------------------------------------

WEIGHTS = {
    "material":    3.0,   # w1: regular piece advantage
    "king":        5.0,   # w2: king advantage
    "mobility":    0.1,   # w3: legal move count advantage
    "promotion":   0.5,   # w4: proximity to promotion rank
    "center":      0.3,   # w5: central square occupancy
}

# Central squares
CENTER_SQUARES = {(3, 2), (3, 4), (4, 1), (4, 3), (4, 5), (3, 3), (4, 2), (4, 4)}

# ---------------------------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------------------------

def evaluate(board: Board, player: str) -> float:
    """
    h(s) = w1*M + w2*K + w3*L + w4*P + w5*C

    Positive = advantageous for `player`.
    """
    opponent = BLACK if player == RED else RED

    my_pieces  = board.pieces_of(player)
    opp_pieces = board.pieces_of(opponent)

    # M — material
    M = len(my_pieces) - len(opp_pieces)

    # K — kings
    my_kings  = sum(1 for _, _, p in my_pieces  if board.is_king(p))
    opp_kings = sum(1 for _, _, p in opp_pieces if board.is_king(p))
    K = my_kings - opp_kings

    # L — mobility
    L = len(board.get_legal_moves(player)) - len(board.get_legal_moves(opponent))

    # P — promotion potential
    def promo(pieces, color):
        score = 0
        for r, _, p in pieces:
            if board.is_king(p):
                continue
            score += max(0, 3 - r) * 0.5 if color == RED else max(0, r - 4) * 0.5
        return score

    P = promo(my_pieces, player) - promo(opp_pieces, opponent)

    # C — center control
    C = (sum(1 for r, c, _ in my_pieces  if (r, c) in CENTER_SQUARES) -
          sum(1 for r, c, _ in opp_pieces if (r, c) in CENTER_SQUARES))

    return (WEIGHTS["material"]  * M +
            WEIGHTS["king"]      * K +
            WEIGHTS["mobility"]  * L +
            WEIGHTS["promotion"] * P +
            WEIGHTS["center"]    * C)

# ---------------------------------------------------------------------------
# Baseline Agent
# ---------------------------------------------------------------------------

class BaselineAgent:
    """
    Minimax with alpha-beta pruning at a fixed depth.
    No move ordering, no transposition table.
    """

    def __init__(
        self,
        player: str,
        depth: int = 5,
        time_limit: float = 5.0,
        rng: Optional[random.Random] = None,
        tie_break_eps: float = 1e-9,
    ):
        self.player     = player
        self.depth      = depth
        self.time_limit = time_limit
        self.rng        = rng
        self.tie_break_eps = tie_break_eps

        self.nodes_expanded = 0
        self.start_time     = 0.0

    def choose_move(self, board: Board) -> Optional[Move]:
        """Select the best move for the current board state."""
        legal_moves = board.get_legal_moves(self.player)
        if not legal_moves:
            return None

        self.nodes_expanded = 0
        self.start_time     = time.time()

        best_move  = None
        best_score = float("-inf")
        best_moves: list[Move] = []

        for move in legal_moves:
            score = self._minimax(
                board.apply_move(move),
                depth=self.depth - 1,
                alpha=float("-inf"),
                beta=float("inf"),
                maximizing=False,
            )
            if score > best_score + self.tie_break_eps:
                best_score = score
                best_moves = [move]
            elif abs(score - best_score) <= self.tie_break_eps:
                best_moves.append(move)

        if best_moves:
            if self.rng is not None:
                best_move = self.rng.choice(best_moves)
            else:
                best_move = best_moves[0]

        return best_move

    def _minimax(self, board: Board, depth: int,
                  alpha: float, beta: float, maximizing: bool) -> float:
        self.nodes_expanded += 1

        if time.time() - self.start_time >= self.time_limit:
            return evaluate(board, self.player)

        terminal_val = board.terminal_score(self.player)
        if terminal_val is not None:
            return terminal_val

        if depth == 0:
            return evaluate(board, self.player)

        current_player = board.current_player
        legal_moves    = board.get_legal_moves(current_player)

        if not legal_moves:
            return LOSS_SCORE if current_player == self.player else WIN_SCORE

        if maximizing:
            value = float("-inf")
            for move in legal_moves:
                value = max(value, self._minimax(
                    board.apply_move(move), depth - 1, alpha, beta, False))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = float("inf")
            for move in legal_moves:
                value = min(value, self._minimax(
                    board.apply_move(move), depth - 1, alpha, beta, True))
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

# ---------------------------------------------------------------------------
# Random Agent
# ---------------------------------------------------------------------------

class RandomAgent:
    """Selects a uniformly random legal move. Used as a control opponent. Woohoo random moves!"""

    def __init__(self, player: str, rng: Optional[random.Random] = None):
        self.player         = player
        self.nodes_expanded = 0
        self.rng            = rng

    def choose_move(self, board: Board) -> Optional[Move]:
        moves = board.get_legal_moves(self.player)
        if not moves:
            return None
        if self.rng is not None:
            return self.rng.choice(moves)
        return random.choice(moves)
