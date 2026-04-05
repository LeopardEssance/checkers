"""
Engine correctness tests (pytest).

Run: pytest tests/
"""

import pytest
from src.engine.board import (
    Board, Move, RED, BLACK,
    RED_PIECE, RED_KING, BLACK_PIECE, BLACK_KING, EMPTY,
    WIN_SCORE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def empty_board():
    b = Board()
    for r in range(8):
        for c in range(8):
            b.grid[r][c] = EMPTY
    return b


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_piece_counts(self):
        board = Board()
        assert board.count_pieces(RED)   == 12
        assert board.count_pieces(BLACK) == 12

    def test_starting_player(self):
        assert Board().current_player == RED

    def test_pieces_on_dark_squares_only(self):
        board = Board()
        for r in range(8):
            for c in range(8):
                if (r + c) % 2 == 0:  # light square
                    assert board.grid[r][c] == EMPTY


class TestLegalMoves:
    def test_opening_move_count(self):
        # RED has exactly 7 opening moves in standard American Checkers
        board = Board()
        assert len(board.get_legal_moves(RED)) == 7

    def test_forced_capture(self, empty_board):
        """If a capture is available it must be the only type of move returned."""
        empty_board.grid[5][2] = RED_PIECE
        empty_board.grid[4][3] = BLACK_PIECE
        empty_board.current_player = RED

        moves = empty_board.get_legal_moves(RED)
        assert len(moves) >= 1
        assert all(m.is_capture for m in moves)

    def test_multi_jump(self, empty_board):
        """A two-capture sequence should appear as a single move."""
        empty_board.grid[6][0] = RED_PIECE
        empty_board.grid[5][1] = BLACK_PIECE
        empty_board.grid[3][3] = BLACK_PIECE
        empty_board.current_player = RED

        moves = empty_board.get_legal_moves(RED)
        two_captures = [m for m in moves if len(m.captures) == 2]
        assert two_captures, "Expected a 2-capture multi-jump sequence"

    def test_jump_stops_after_promotion(self, empty_board):
        """A man that crowns during capture cannot continue jumping this turn."""
        empty_board.grid[2][1] = RED_PIECE
        empty_board.grid[1][2] = BLACK_PIECE
        # This capture would be possible only if RED were allowed to continue as a king.
        empty_board.grid[1][4] = BLACK_PIECE
        empty_board.current_player = RED

        moves = empty_board.get_legal_moves(RED)
        assert moves
        assert all(m.is_capture for m in moves)
        assert all(len(m.captures) == 1 for m in moves)
        assert any(m.destination == (0, 3) for m in moves)

    def test_king_moves_backwards(self, empty_board):
        """King pieces can move in all four diagonal directions."""
        empty_board.grid[4][4] = RED_KING
        empty_board.current_player = RED

        destinations = {m.destination for m in empty_board.get_legal_moves(RED)}
        # Should include both forward and backward destinations
        assert (3, 3) in destinations
        assert (3, 5) in destinations
        assert (5, 3) in destinations
        assert (5, 5) in destinations


class TestMoveApplication:
    def test_player_switches(self):
        board = Board()
        move  = board.get_legal_moves(RED)[0]
        new   = board.apply_move(move)
        assert new.current_player == BLACK

    def test_piece_moves_to_destination(self, empty_board):
        empty_board.grid[5][2] = RED_PIECE
        empty_board.current_player = RED
        move = empty_board.get_legal_moves(RED)[0]
        new  = empty_board.apply_move(move)
        assert new.grid[move.destination[0]][move.destination[1]] == RED_PIECE
        assert new.grid[5][2] == EMPTY

    def test_capture_removes_piece(self, empty_board):
        empty_board.grid[5][2] = RED_PIECE
        empty_board.grid[4][3] = BLACK_PIECE
        empty_board.current_player = RED
        moves   = empty_board.get_legal_moves(RED)
        capture = next(m for m in moves if m.is_capture)
        new     = empty_board.apply_move(capture)
        assert new.grid[4][3] == EMPTY

    def test_promotion_red(self, empty_board):
        empty_board.grid[1][2] = RED_PIECE
        empty_board.current_player = RED
        promo = next(m for m in empty_board.get_legal_moves(RED)
                    if m.destination[0] == 0)
        new = empty_board.apply_move(promo)
        assert new.grid[0][promo.destination[1]] == RED_KING

    def test_promotion_black(self, empty_board):
        empty_board.grid[6][3] = BLACK_PIECE
        empty_board.current_player = BLACK
        promo = next(m for m in empty_board.get_legal_moves(BLACK)
                    if m.destination[0] == 7)
        new = empty_board.apply_move(promo)
        assert new.grid[7][promo.destination[1]] == BLACK_KING

    def test_immutability(self):
        """apply_move must not mutate the original board."""
        board = Board()
        orig  = str(board)
        move  = board.get_legal_moves(RED)[0]
        board.apply_move(move)
        assert str(board) == orig


class TestTerminalDetection:
    def test_no_black_pieces_is_terminal(self, empty_board):
        empty_board.grid[3][2] = RED_PIECE
        assert empty_board.is_terminal()

    def test_red_wins_when_black_has_no_pieces(self, empty_board):
        empty_board.grid[3][2] = RED_PIECE
        assert empty_board.terminal_score(RED) == WIN_SCORE

    def test_black_wins_when_red_has_no_pieces(self, empty_board):
        empty_board.grid[3][2] = BLACK_PIECE
        empty_board.current_player = BLACK
        assert empty_board.terminal_score(BLACK) == WIN_SCORE

    def test_opening_is_not_terminal(self):
        assert not Board().is_terminal()
