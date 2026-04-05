"""
State representation (B, P):
  B = 8x8 board grid (32 playable dark squares)
  P = player to move (RED or BLACK)

Implements: piece movement, forced-capture rule, multi-jump sequences,
king promotion, and terminal-state detection.
"""

from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RED   = "R"
BLACK = "B"

EMPTY       = "."
RED_PIECE   = "r"
RED_KING    = "R"
BLACK_PIECE = "b"
BLACK_KING  = "B"

WIN_SCORE  =  10_000
LOSS_SCORE = -10_000
DRAW_SCORE =  0


# ---------------------------------------------------------------------------
# Move
# ---------------------------------------------------------------------------

@dataclass
class Move:
    """
    A single action — possibly a multi-jump sequence.

    path     : squares visited from origin to final landing square.
    captures : squares of enemy pieces removed during the action.
    """
    path: List[Tuple[int, int]]
    captures: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def origin(self) -> Tuple[int, int]:
        return self.path[0]

    @property
    def destination(self) -> Tuple[int, int]:
        return self.path[-1]

    @property
    def is_capture(self) -> bool:
        return len(self.captures) > 0

    def __repr__(self) -> str:
        arrow = " -> ".join(str(s) for s in self.path)
        cap   = f"  captures={self.captures}" if self.captures else ""
        return f"Move({arrow}{cap})"


# ---------------------------------------------------------------------------
# Board
# ---------------------------------------------------------------------------

class Board:
    """
    Full game state (B, P).

    The 8x8 grid is stored as a list-of-lists.
    Only dark squares (row + col odd) are ever occupied.
    """

    def __init__(self):
        self.grid: List[List[str]] = [[EMPTY] * 8 for _ in range(8)]
        self.current_player: str = RED  # RED moves first (fixed for experiment reproducibility)
        self._setup_pieces()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_pieces(self):
        """Standard American Checkers opening: 12 pieces per player."""
        for row in range(8):
            for col in range(8):
                if (row + col) % 2 == 1:
                    if row < 3:
                        self.grid[row][col] = BLACK_PIECE
                    elif row > 4:
                        self.grid[row][col] = RED_PIECE

    # ------------------------------------------------------------------
    # Piece helpers
    # ------------------------------------------------------------------

    def get(self, row: int, col: int) -> str:
        return self.grid[row][col]

    def set(self, row: int, col: int, piece: str):
        self.grid[row][col] = piece

    @staticmethod
    def is_red(piece: str) -> bool:
        return piece in (RED_PIECE, RED_KING)

    @staticmethod
    def is_black(piece: str) -> bool:
        return piece in (BLACK_PIECE, BLACK_KING)

    @staticmethod
    def is_king(piece: str) -> bool:
        return piece in (RED_KING, BLACK_KING)

    @staticmethod
    def owner(piece: str) -> Optional[str]:
        if piece in (RED_PIECE, RED_KING):
            return RED
        if piece in (BLACK_PIECE, BLACK_KING):
            return BLACK
        return None

    def is_opponent(self, piece: str, player: str) -> bool:
        o = self.owner(piece)
        return o is not None and o != player

    def pieces_of(self, player: str) -> List[Tuple[int, int, str]]:
        """Return [(row, col, piece)] for all pieces belonging to player."""
        return [
            (r, c, self.grid[r][c])
            for r in range(8) for c in range(8)
            if self.owner(self.grid[r][c]) == player
        ]

    def count_pieces(self, player: str) -> int:
        return len(self.pieces_of(player))

    # ------------------------------------------------------------------
    # Move generation
    # ------------------------------------------------------------------

    def _move_dirs(self, piece: str) -> List[Tuple[int, int]]:
        if piece == RED_PIECE:
            return [(-1, -1), (-1, 1)]
        if piece == BLACK_PIECE:
            return [(1, -1), (1, 1)]
        return [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # king

    def _get_jumps(
        self,
        row: int, col: int,
        piece: str, player: str,
        visited_captures: List[Tuple[int, int]],
        path: List[Tuple[int, int]],
    ) -> List[Move]:
        """Recursively generate all multi-jump sequences from (row, col)."""
        found: List[Move] = []

        for dr, dc in self._move_dirs(piece):
            mid_r, mid_c   = row + dr,     col + dc
            land_r, land_c = row + 2 * dr, col + 2 * dc

            if not (0 <= mid_r < 8 and 0 <= mid_c < 8):
                continue
            if not (0 <= land_r < 8 and 0 <= land_c < 8):
                continue
            if not self.is_opponent(self.grid[mid_r][mid_c], player):
                continue
            if self.grid[land_r][land_c] != EMPTY:
                continue
            if (mid_r, mid_c) in visited_captures:
                continue

            new_path     = path + [(land_r, land_c)]
            new_captures = visited_captures + [(mid_r, mid_c)]

            # In American checkers, a move ends immediately on crowning.
            promoted = piece
            if piece == RED_PIECE   and land_r == 0: promoted = RED_KING
            if piece == BLACK_PIECE and land_r == 7: promoted = BLACK_KING

            if promoted != piece:
                found.append(Move(path=new_path, captures=new_captures))
                continue

            continuations = self._get_jumps(
                land_r, land_c, promoted, player, new_captures, new_path
            )
            if continuations:
                found.extend(continuations)
            else:
                found.append(Move(path=new_path, captures=new_captures))

        return found

    def get_legal_moves(self, player: Optional[str] = None, enforce_capture: bool = True) -> List[Move]:
        """
        All legal moves for player (defaults to current_player).
        Forced-capture rule: if any capture exists, only captures are returned
        (unless enforce_capture=False, only set for human UI players).
        """
        if player is None:
            player = self.current_player

        captures: List[Move] = []
        simple:   List[Move] = []

        for row, col, piece in self.pieces_of(player):
            captures.extend(self._get_jumps(row, col, piece, player, [], [(row, col)]))
            for dr, dc in self._move_dirs(piece):
                nr, nc = row + dr, col + dc
                if 0 <= nr < 8 and 0 <= nc < 8 and self.grid[nr][nc] == EMPTY:
                    simple.append(Move(path=[(row, col), (nr, nc)]))

        if enforce_capture: # this is for allowing a player to play later down the line
            return captures if captures else simple
        else:
            return captures + simple

    # ------------------------------------------------------------------
    # Apply move
    # ------------------------------------------------------------------

    def apply_move(self, move: Move) -> "Board":
        """Return a new Board with move applied (does not mutate self)."""
        new = deepcopy(self)
        origin, dest = move.origin, move.destination
        piece = new.grid[origin[0]][origin[1]]

        new.grid[origin[0]][origin[1]] = EMPTY
        new.grid[dest[0]][dest[1]]     = piece

        for cr, cc in move.captures:
            new.grid[cr][cc] = EMPTY

        if piece == RED_PIECE   and dest[0] == 0: new.grid[dest[0]][dest[1]] = RED_KING
        if piece == BLACK_PIECE and dest[0] == 7: new.grid[dest[0]][dest[1]] = BLACK_KING

        new.current_player = BLACK if new.current_player == RED else RED
        return new

    # ------------------------------------------------------------------
    # Terminal detection
    # ------------------------------------------------------------------

    def _terminal_result(self) -> Optional[str]:
        """
        Returns 'R' (RED wins), 'B' (BLACK wins), 'DRAW', or None (ongoing).
        """
        red_pieces   = self.count_pieces(RED)
        black_pieces = self.count_pieces(BLACK)

        if red_pieces   == 0: return BLACK
        if black_pieces == 0: return RED

        red_moves   = self.get_legal_moves(RED)
        black_moves = self.get_legal_moves(BLACK)

        if not red_moves and not black_moves: return "DRAW"
        if not red_moves:   return BLACK
        if not black_moves: return RED

        return None

    def is_terminal(self) -> bool:
        return self._terminal_result() is not None

    def terminal_score(self, maximizing_player: str) -> Optional[float]:
        """WIN/LOSS/DRAW score from maximizing_player's perspective, or None."""
        result = self._terminal_result()
        if result is None:   return None
        if result == "DRAW": return DRAW_SCORE
        return WIN_SCORE if result == maximizing_player else LOSS_SCORE

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        lines = ["   " + " ".join(str(c) for c in range(8))]
        lines.append("  +" + "-" * 15 + "+")
        for r in range(8):
            row_str = " ".join(
                self.grid[r][c] if self.grid[r][c] != EMPTY else "."
                for c in range(8)
            )
            lines.append(f"{r} | {row_str} |")
        lines.append("  +" + "-" * 15 + "+")
        lines.append(f"  Current player: {self.current_player}")
        return "\n".join(lines)

    def copy(self) -> "Board":
        return deepcopy(self)
