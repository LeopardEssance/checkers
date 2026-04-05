"""
Enhancement #1: Move Ordering

Enhancements over baseline:
  - Killer Heuristic: moves that caused beta-cutoffs at a given depth ply
    are tried first at the same ply on future calls.
  - History Heuristic: accumulates a score per (origin, destination) pair
    each time that move causes a cutoff; higher scores are explored first.
  - Static capture-first ordering as a lightweight first pass.

Both heuristics can be toggled independently for ablation.
"""

import time
from collections import defaultdict
from typing import Optional, List, Dict, Tuple

from src.engine.board import Board, Move, RED, BLACK, WIN_SCORE, LOSS_SCORE
from src.agents.base  import evaluate

class MoveOrderingAgent:
    """
    Minimax + Alpha-Beta + configurable move ordering.

    Parameters
    ----------
    use_killer  : enable Killer Heuristic
    use_history : enable History Heuristic
    """

    MAX_KILLERS = 2

    def __init__(
        self,
        player:      str,
        depth:       int   = 5,
        time_limit:  float = 5.0,
        use_killer:  bool  = True,
        use_history: bool  = True,
    ):
        self.player      = player
        self.depth       = depth
        self.time_limit  = time_limit
        self.use_killer  = use_killer
        self.use_history = use_history

        # Killer table: killer_table[ply] = [Move, ...]
        self.killer_table:  Dict[int, List[Move]] = defaultdict(list)
        # History table: (origin, dest) -> cumulative cutoff score
        self.history_table: Dict[Tuple, int]      = defaultdict(int)

        self.nodes_expanded = 0
        self.start_time     = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_killers(self):
        self.killer_table = defaultdict(list)

    def _store_killer(self, ply: int, move: Move):
        killers = self.killer_table[ply]
        for k in killers:
            if k.origin == move.origin and k.destination == move.destination:
                return
        killers.insert(0, move)
        if len(killers) > self.MAX_KILLERS:
            killers.pop()

    def _killer_score(self, ply: int, move: Move) -> int:
        for i, k in enumerate(self.killer_table.get(ply, [])):
            if k.origin == move.origin and k.destination == move.destination:
                return self.MAX_KILLERS - i
        return 0

    def _order_moves(self, moves: List[Move], ply: int) -> List[Move]:
        """Sort moves by combined priority (descending)."""
        def key(m: Move):
            cap     = len(m.captures) * 10_000
            killer  = self._killer_score(ply, m) * 1_000 if self.use_killer  else 0
            history = self.history_table.get((m.origin, m.destination), 0) \
                      if self.use_history else 0
            return -(cap + killer + history)
        return sorted(moves, key=key)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def choose_move(self, board: Board) -> Optional[Move]:
        """Select the best move for the current board state."""
        legal_moves = board.get_legal_moves(self.player)
        if not legal_moves:
            return None

        self._reset_killers()
        self.nodes_expanded = 0
        self.start_time     = time.time()

        best_move  = None
        best_score = float("-inf")
        alpha, beta = float("-inf"), float("inf")

        for move in self._order_moves(legal_moves, ply=0):
            score = self._minimax(board.apply_move(move),
                                  depth=self.depth - 1,
                                  alpha=alpha, beta=beta,
                                  maximizing=False, ply=1)
            if score > best_score:
                best_score = score
                best_move  = move
            alpha = max(alpha, score)

        return best_move

    def _minimax(self, board: Board, depth: int, 
                  alpha: float, beta: float,
                  maximizing: bool, ply: int) -> float:
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

        ordered = self._order_moves(legal_moves, ply)

        if maximizing:
            value = float("-inf")
            for move in ordered:
                value = max(value, self._minimax(
                    board.apply_move(move), depth - 1, alpha, beta, False, ply + 1))
                alpha = max(alpha, value)
                if alpha >= beta:
                    if self.use_killer:  self._store_killer(ply, move)
                    if self.use_history:
                        self.history_table[(move.origin, move.destination)] += 2 ** depth
                    break
            return value
        else:
            value = float("inf")
            for move in ordered:
                value = min(value, self._minimax(
                    board.apply_move(move), depth - 1, alpha, beta, True, ply + 1))
                beta = min(beta, value)
                if beta <= alpha:
                    if self.use_killer:  self._store_killer(ply, move)
                    if self.use_history:
                        self.history_table[(move.origin, move.destination)] += 2 ** depth
                    break
            return value
