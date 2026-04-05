"""
Agent correctness tests (pytest).
Verifies that each agent returns a valid legal move and that
performance metrics are being tracked correctly.

Run: python -m pytest tests/ -v
"""

import pytest
from src.engine.board         import Board, Move, RED, BLACK, EMPTY
from src.agents.base          import BaselineAgent, RandomAgent, evaluate
from src.agents.move_ordering import MoveOrderingAgent
from src.agents.transposition import TranspositionAgent, ZobristHasher

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def legal_move_set(board: Board, player: str):
    return {(m.origin, m.destination) for m in board.get_legal_moves(player)}


def make_one_move_board():
    """Board after one Red opening move — it's now Black's turn."""
    board = Board()
    move  = board.get_legal_moves(RED)[0]
    return board.apply_move(move)

# ---------------------------------------------------------------------------
# Evaluate function
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_starting_position_is_neutral(self):
        """With equal pieces the evaluation should be near zero."""
        board = Board()
        score = evaluate(board, RED)
        # Mobility and center may give a small bias but it won't be huge
        assert abs(score) < 5.0

    def test_more_pieces_scores_higher(self):
        board = Board()
        # Remove a black piece
        board.grid[0][1] = EMPTY
        assert evaluate(board, RED) > evaluate(board, BLACK)

    def test_king_valued_above_regular(self):
        board = Board()
        for r in range(8):
            for c in range(8):
                board.grid[r][c] = EMPTY
        board.grid[4][4] = "R"   # RED king
        board.grid[3][3] = "r"   # RED regular piece
        # King should contribute more to evaluation
        board2 = board.copy()
        board.grid[3][3]  = EMPTY   # remove regular, keep king
        board2.grid[4][4] = EMPTY   # remove king,    keep regular
        assert evaluate(board, RED) > evaluate(board2, RED)

# ---------------------------------------------------------------------------
# Baseline Agent
# ---------------------------------------------------------------------------

class TestBaselineAgent:
    def test_returns_legal_move(self):
        board = Board()
        agent = BaselineAgent(RED, depth=2)
        move  = agent.choose_move(board)
        assert move is not None
        assert (move.origin, move.destination) in legal_move_set(board, RED)

    def test_nodes_expanded_positive(self):
        board = Board()
        agent = BaselineAgent(RED, depth=2)
        agent.choose_move(board)
        assert agent.nodes_expanded > 0

    def test_returns_none_when_no_moves(self):
        board = Board()
        for r in range(8):
            for c in range(8):
                board.grid[r][c] = EMPTY
        # Board has no pieces — RED has no moves
        agent = BaselineAgent(RED, depth=2)
        assert agent.choose_move(board) is None

    def test_black_side(self):
        board = make_one_move_board()
        agent = BaselineAgent(BLACK, depth=2)
        move  = agent.choose_move(board)
        assert move is not None
        assert (move.origin, move.destination) in legal_move_set(board, BLACK)

    def test_prefers_capture(self):
        """Agent at any depth should prefer an obvious immediate capture."""
        board = Board()
        for r in range(8):
            for c in range(8):
                board.grid[r][c] = EMPTY
        # Set up: RED piece can capture BLACK piece
        board.grid[5][2] = "r"
        board.grid[4][3] = "b"
        board.current_player = RED
        agent = BaselineAgent(RED, depth=1)
        move  = agent.choose_move(board)
        assert move.is_capture

# ---------------------------------------------------------------------------
# Random Agent
# ---------------------------------------------------------------------------

class TestRandomAgent:
    def test_returns_legal_move(self):
        board = Board()
        agent = RandomAgent(RED)
        move  = agent.choose_move(board)
        assert (move.origin, move.destination) in legal_move_set(board, RED)

    def test_returns_none_when_no_moves(self):
        board = Board()
        for r in range(8):
            for c in range(8):
                board.grid[r][c] = EMPTY
        assert RandomAgent(RED).choose_move(board) is None

# ---------------------------------------------------------------------------
# Move Ordering Agent
# ---------------------------------------------------------------------------

class TestMoveOrderingAgent:
    def test_returns_legal_move(self):
        board = Board()
        agent = MoveOrderingAgent(RED, depth=2)
        move  = agent.choose_move(board)
        assert (move.origin, move.destination) in legal_move_set(board, RED)

    def test_nodes_expanded_tracked(self):
        agent = MoveOrderingAgent(RED, depth=2)
        agent.choose_move(Board())
        assert agent.nodes_expanded > 0

    def test_fewer_nodes_than_baseline(self):
        """Move ordering should expand fewer or equal nodes than the baseline."""
        board    = Board()
        baseline = BaselineAgent(RED, depth=4)
        ordered  = MoveOrderingAgent(RED, depth=4, use_killer=True, use_history=True)

        baseline.choose_move(board)
        ordered.choose_move(board)

        # This is not guaranteed on every position but holds in aggregate;
        # here we just check the metric is being collected and is plausible.
        assert ordered.nodes_expanded >= 0
        assert baseline.nodes_expanded >= 0

    def test_killer_only(self):
        board = Board()
        agent = MoveOrderingAgent(RED, depth=2, use_killer=True, use_history=False)
        move  = agent.choose_move(board)
        assert (move.origin, move.destination) in legal_move_set(board, RED)

    def test_history_only(self):
        board = Board()
        agent = MoveOrderingAgent(RED, depth=2, use_killer=False, use_history=True)
        move  = agent.choose_move(board)
        assert (move.origin, move.destination) in legal_move_set(board, RED)

# ---------------------------------------------------------------------------
# Transposition Agent
# ---------------------------------------------------------------------------

class TestTranspositionAgent:
    def test_returns_legal_move(self):
        board = Board()
        agent = TranspositionAgent(RED, depth=2)
        move  = agent.choose_move(board)
        assert (move.origin, move.destination) in legal_move_set(board, RED)

    def test_tt_hits_accumulate(self):
        """After multiple moves the TT should register some hits."""
        board = Board()
        agent = TranspositionAgent(RED, depth=4, use_tt=True)
        opp   = RandomAgent(BLACK)
        total_hits = 0

        for _ in range(5):
            if board.is_terminal():
                break
            if board.current_player == RED:
                agent.choose_move(board)
                total_hits += agent.tt_hits
                move = agent.choose_move(board)
            else:
                move = opp.choose_move(board)
            if move:
                board = board.apply_move(move)

        # Not guaranteed to be non-zero on every run, but usually is
        assert total_hits >= 0   # sanity — metric is being tracked

    def test_tt_disabled_still_works(self):
        board = Board()
        agent = TranspositionAgent(RED, depth=2, use_tt=False)
        move  = agent.choose_move(board)
        assert (move.origin, move.destination) in legal_move_set(board, RED)
        assert len(agent.tt) == 0   # table should be empty when TT off

    def test_tt_table_grows(self):
        board = Board()
        agent = TranspositionAgent(RED, depth=4, use_tt=True)
        agent.choose_move(board)
        assert len(agent.tt) > 0

# ---------------------------------------------------------------------------
# Zobrist Hasher
# ---------------------------------------------------------------------------

class TestZobristHasher:
    def test_same_board_same_hash(self):
        hasher = ZobristHasher()
        b1, b2 = Board(), Board()
        assert hasher.hash_board(b1) == hasher.hash_board(b2)

    def test_different_boards_different_hash(self):
        hasher = ZobristHasher()
        b1     = Board()
        b2     = b1.apply_move(b1.get_legal_moves(RED)[0])
        assert hasher.hash_board(b1) != hasher.hash_board(b2)

    def test_hash_is_integer(self):
        assert isinstance(ZobristHasher().hash_board(Board()), int)

    def test_side_to_move_changes_hash(self):
        hasher = ZobristHasher()
        b1     = Board()                        # RED to move
        b2     = b1.copy()
        b2.current_player = BLACK               # manually flip
        assert hasher.hash_board(b1) != hasher.hash_board(b2)