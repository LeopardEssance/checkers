"""
Enhancement #2: Transposition Tables (Zobrist Hashing)

Implements:
  - Zobrist hashing: random 64-bit key per (square, piece-type) pair,
    XOR-composed to form a unique board fingerprint.
  - Transposition table: cache (hash → TTEntry{score, depth, flag, best_move}).
    Flags: EXACT, LOWERBOUND, UPPERBOUND — correct integration with alpha-beta.
  - Full enhanced agent combining TT + Killer + History move ordering.

"""

import time
import random
from collections import defaultdict
from enum import IntEnum
from typing import Optional, Dict

from src.engine.board import Board, Move, RED, BLACK, RED_PIECE, RED_KING, BLACK_PIECE, BLACK_KING, EMPTY, WIN_SCORE, LOSS_SCORE
from src.agents.base import evaluate

# ---------------------------------------------------------------------------
# Transposition table entry flags
# ---------------------------------------------------------------------------

class TTFlag(IntEnum):
    EXACT      = 0
    LOWERBOUND = 1
    UPPERBOUND = 2

# ---------------------------------------------------------------------------
# Zobrist Hasher
# ---------------------------------------------------------------------------

PIECE_TYPES = [RED_PIECE, RED_KING, BLACK_PIECE, BLACK_KING]

class ZobristHasher:
    """
    Zobrist hashing: Fast, deterministic board state fingerprints.
    
    Each (row, col, piece_type) gets a random 64-bit key. Board hash is computed
    by XORing keys of all occupied squares, with an additional bit for side-to-move.
    
    Collisions are extremely rare (virtually impossible for different positions).
    Used by transposition table to quickly check if a position has been seen before.
    """
    def __init__(self, seed: int = 12345):
        rng = random.Random(seed)
        self.piece_keys = [
            [{p: rng.getrandbits(64) for p in PIECE_TYPES} for _ in range(8)]
            for _ in range(8)
        ]
        self.black_to_move_key = rng.getrandbits(64)

    def hash_board(self, board: Board) -> int:
        """Compute full hash from scratch by XORing all occupied-square keys."""
        h = 0
        for r in range(8):
            for c in range(8):
                piece = board.grid[r][c]
                if piece != EMPTY:
                    h ^= self.piece_keys[r][c][piece]
        if board.current_player == BLACK:
            h ^= self.black_to_move_key
        return h

# ---------------------------------------------------------------------------
# TT Entry
# ---------------------------------------------------------------------------

class TTEntry:
    """
    Transposition table entry: stores result of a minimax search at a position.
    
    Attributes:
      score     : the value returned by that search
      depth     : search depth at which this value was computed
      flag      : indicates meaning of score relative to alpha-beta window:
                  - EXACT: score is exact within [alpha, beta]
                  - LOWERBOUND: score >= beta (a cutoff occurred)
                  - UPPERBOUND: score <= alpha (all moves were worse)
      best_move : the best move found in that search (for move ordering)
    """
    __slots__ = ("score", "depth", "flag", "best_move")

    def __init__(self, score: float, depth: int, flag: TTFlag,
                  best_move: Optional[Move]):
        self.score     = score
        self.depth     = depth
        self.flag      = flag
        self.best_move = best_move

# ---------------------------------------------------------------------------
# Full Enhanced Agent
# ---------------------------------------------------------------------------

class TranspositionAgent:
    """
    Minimax + Alpha-Beta + Move Ordering (Killer + History) + Transposition Table.

    This is the complete M2 agent described in the project proposal.
    The transposition table can be toggled off for ablation.
    """

    MAX_TT_SIZE = 1_000_000
    MAX_KILLERS = 2

    def __init__(
        self,
        player:     str,
        depth:      int   = 5,
        time_limit: float = 5.0,
        use_tt:     bool  = True,
        rng: Optional[random.Random] = None,
        tie_break_eps: float = 1e-9,
    ):
        self.player     = player
        self.depth      = depth
        self.time_limit = time_limit
        self.use_tt     = use_tt
        self.rng        = rng
        self.tie_break_eps = tie_break_eps

        self.hasher = ZobristHasher()
        self.tt:    Dict[int, TTEntry] = {}

        self.killer_table  = defaultdict(list)
        self.history_table = defaultdict(int)

        self.nodes_expanded = 0
        self.tt_hits        = 0
        self.start_time     = 0.0

    # ------------------------------------------------------------------
    # Move ordering helpers
    # ------------------------------------------------------------------

    def _store_killer(self, ply: int, move: Move):
        killers = self.killer_table[ply]
        for k in killers:
            if k.origin == move.origin and k.destination == move.destination:
                return
        killers.insert(0, move)
        if len(killers) > self.MAX_KILLERS:
            killers.pop()

    def _order_moves(self, moves, ply: int, tt_best: Optional[Move]) -> list:
        def key(m: Move):
            if (tt_best
                    and m.origin      == tt_best.origin
                    and m.destination == tt_best.destination):
                return -1_000_000_000
            cap     = len(m.captures) * 10_000
            killer  = 0
            history = self.history_table.get((m.origin, m.destination), 0)
            for i, k in enumerate(self.killer_table.get(ply, [])):
                if k.origin == m.origin and k.destination == m.destination:
                    killer = (self.MAX_KILLERS - i) * 1_000
                    break
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

        self.killer_table   = defaultdict(list)
        self.nodes_expanded = 0
        self.tt_hits        = 0
        self.start_time     = time.time()

        best_move  = None
        best_score = float("-inf")
        best_moves: list[Move] = []
        alpha, beta = float("-inf"), float("inf")

        board_hash = self.hasher.hash_board(board)
        tt_best    = self.tt[board_hash].best_move if self.use_tt and board_hash in self.tt else None

        for move in self._order_moves(legal_moves, ply=0, tt_best=tt_best):
            score = self._minimax(board.apply_move(move),
                                  depth=self.depth - 1,
                                  alpha=alpha, beta=beta,
                                  maximizing=False, ply=1)
            if score > best_score + self.tie_break_eps:
                best_score = score
                best_moves = [move]
            elif abs(score - best_score) <= self.tie_break_eps:
                best_moves.append(move)
            alpha = max(alpha, score)

        if best_moves:
            if self.rng is not None:
                best_move = self.rng.choice(best_moves)
            else:
                best_move = best_moves[0]

        return best_move

    def _minimax(self, board: Board, depth: int, alpha: float, beta: float,
                  maximizing: bool, ply: int) -> float:
        """
        Minimax with transposition table (TT) integration.
        
        Flow:
          1. Increment node count and check time limit
          2. Check TT for existing entry at this position and depth:
             - EXACT flag: return score directly
             - LOWERBOUND: tighten alpha (improve guarantee for maximizer)
             - UPPERBOUND: tighten beta (improve guarantee for minimizer)
             - If alpha >= beta after tightening, pruning occurs
          3. Recurse with alpha-beta pruning
          4. Store result in TT with appropriate flag based on final value:
             - EXACT if value is within [orig_alpha, orig_beta]
             - LOWERBOUND if value >= orig_beta (minimizer couldn't prevent this)
             - UPPERBOUND if value <= orig_alpha (maximizer couldn't guarantee this)
        """
        self.nodes_expanded += 1
        orig_alpha = alpha
        orig_beta = beta

        if time.time() - self.start_time >= self.time_limit:
            return evaluate(board, self.player)

        terminal_val = board.terminal_score(self.player)
        if terminal_val is not None:
            return terminal_val

        board_hash = self.hasher.hash_board(board) if self.use_tt else None
        tt_best    = None

        # === TT lookup ===
        if self.use_tt and board_hash in self.tt:
            entry = self.tt[board_hash]
            # Only use TT entry if it was computed at sufficient depth
            if entry.depth >= depth:
                self.tt_hits += 1
                if   entry.flag == TTFlag.EXACT:      return entry.score
                elif entry.flag == TTFlag.LOWERBOUND: alpha = max(alpha, entry.score)
                elif entry.flag == TTFlag.UPPERBOUND: beta  = min(beta,  entry.score)
                if alpha >= beta:
                    return entry.score  # Pruning via TT
            tt_best = entry.best_move

        if depth == 0:
            score = evaluate(board, self.player)
            if self.use_tt:
                self._tt_store(board_hash, score, 0, TTFlag.EXACT, None)
            return score

        current_player = board.current_player
        legal_moves    = board.get_legal_moves(current_player)

        if not legal_moves:
            return LOSS_SCORE if current_player == self.player else WIN_SCORE

        best_move = None

        if maximizing:
            value = float("-inf")
            for move in self._order_moves(legal_moves, ply, tt_best):
                s = self._minimax(board.apply_move(move),
                                  depth - 1, alpha, beta, False, ply + 1)
                if s > value:
                    value     = s
                    best_move = move
                alpha = max(alpha, value)
                if alpha >= beta:
                    self._store_killer(ply, move)
                    self.history_table[(move.origin, move.destination)] += 2 ** depth
                    break
        else:
            value = float("inf")
            for move in self._order_moves(legal_moves, ply, tt_best):
                s = self._minimax(board.apply_move(move),
                                  depth - 1, alpha, beta, True, ply + 1)
                if s < value:
                    value     = s
                    best_move = move
                beta = min(beta, value)
                if beta <= alpha:
                    self._store_killer(ply, move)
                    self.history_table[(move.origin, move.destination)] += 2 ** depth
                    break

        # === TT storage ===
        if self.use_tt:
            # Determine entry flag based on final value relative to original window
            if   value <= orig_alpha: flag = TTFlag.UPPERBOUND
            elif value >= orig_beta:  flag = TTFlag.LOWERBOUND
            else:                     flag = TTFlag.EXACT
            self._tt_store(board_hash, value, depth, flag, best_move)

        return value

    def _tt_store(self, h: int, score: float, depth: int,
                  flag: TTFlag, best_move: Optional[Move]):
        """
        Store (or update) a transposition table entry at position hash h.
        
        If table is full, evict an arbitrary old entry (simple replacement strategy).
        Better strategies like LRU or age-based eviction could be used if needed.
        """
        if len(self.tt) >= self.MAX_TT_SIZE:
            self.tt.pop(next(iter(self.tt)))
        self.tt[h] = TTEntry(score, depth, flag, best_move)
