# CS 57200 — Checkers AI Agent

**Lauren Fries | Track A (Game AI) | Heuristic Problem Solving**

A checkers-playing agent built to analyze the incremental impact of adversarial search enhancements on playing strength and computational efficiency. The project implements a Minimax + Alpha-Beta baseline and layers on two enhancements — Move Ordering and Transposition Tables — then measures their effect through controlled experiments.

---

## Project Structure

```
.
├── src/
│   ├── engine/
│   │   ├── board.py            # Game state, legal move generation, terminal detection
│   │   └── game_runner.py      # Plays full games between two agents, tracks stats
│   └── agents/
│       ├── base.py             # BaselineAgent (Minimax + Alpha-Beta) and RandomAgent
│       ├── move_ordering.py    # Enhancement #1: Killer + History heuristics
│       └── transposition.py    # Enhancement #2: Zobrist hashing + Transposition Table
├── experiments/
│   ├── exp0_baseline_vs_random.py          # Sanity check: Baseline vs Random
│   ├── exp0_move_ordering_vs_random.py     # Sanity check: Move Ordering vs Random
│   ├── exp0_transposition_vs_random.py     # Sanity check: Transposition vs Random
│   ├── run_exp0_pipeline.py                # Single entry point for all Exp 0 variants
│   ├── experiment1_head_to_head.py         # Head-to-head agent comparison
│   ├── experiment2_ablation.py             # Ablation study (one component disabled at a time)
│   ├── experiment3_scalability.py          # Depth vs performance scaling
│   └── plot_performance.py                 # Generate charts from JSON result files
├── tests/
│   ├── test_engine.py          # Engine correctness tests (pytest)
│   └── test_agents.py          # Agent correctness tests (pytest)
├── results/                    # Generated experiment output
├── requirements.txt
└── README.md
```

---

## Setup

**Requirements:** Python 3.10+

```bash
# Clone the repo
git clone https://github.com/LeopardEssance/checkers.git
cd checkers

# Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install dependencies (required)
pip install -r requirements.txt
```

⚠️ **Important:** Always run `pip install -r requirements.txt` after creating or activating your virtual environment. The experiments will fail without matplotlib and pytest installed.

**Dependencies** (`requirements.txt`):

```
pytest
matplotlib
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

All tests should pass. The suite covers engine correctness (move generation, forced captures, promotion, terminal detection) and agent correctness (legal move selection, node tracking, TT behavior).

---

## Running Experiments

### Experiment 0 — Sanity Checks (Agent vs. Random)

Use the pipeline runner to run any variant:

```bash
# Baseline vs Random (50 games, depth 4)
python -m experiments.run_exp0_pipeline --run baseline

# Move Ordering vs Random
python -m experiments.run_exp0_pipeline --run move-ordering

# Transposition vs Random
python -m experiments.run_exp0_pipeline --run transposition

# All three variants
python -m experiments.run_exp0_pipeline --run all

# Custom settings
python -m experiments.run_exp0_pipeline --run baseline --games 100 --depth 5

# Skip saving results or graphs
python -m experiments.run_exp0_pipeline --run baseline --no-save --no-plot

# Custom graph output directory and label
python -m experiments.run_exp0_pipeline --run baseline --plot-output-dir ./my_graphs --plot-label "My Dataset"
```

Results are saved to `results/exp0/<variant>/d{depth}_n{games}/`.

---

### Experiment 1 — Head-to-Head Comparison

```bash
python -m experiments.experiment1_head_to_head

# Custom settings
python -m experiments.experiment1_head_to_head --games 25 --depth 4 --seed 42 --stochastic-tiebreak
```

Runs five matchups: Transposition vs Baseline, Transposition vs Move Ordering, and each agent vs Random.

---

### Experiment 2 — Ablation Study

```bash
python -m experiments.experiment2_ablation

# Custom settings
python -m experiments.experiment2_ablation --games 20 --depth 4 --seed 42
```

Tests six configurations, disabling one component at a time (TT, Killer, History, all ordering), all against Random.

---

### Experiment 3 — Scalability (Depth vs. Performance)

```bash
python -m experiments.experiment3_scalability

# Custom depths
python -m experiments.experiment3_scalability --depths 3 4 5 6 --games 10

# Regenerate graphs from existing run without re-running games
python -m experiments.experiment3_scalability --plot-from-run-dir results/experiment3_scalability/d3to6_n200
```

Measures how each configuration scales as search depth increases from 3 to 6.

---

### Experiment-Specific Flags

#### All Experiments

| Flag        | Description                  | Default |
| ----------- | ---------------------------- | ------- |
| `--verbose` | Print board state each move  | Off     |
| `--no-save` | Skip writing JSON results    | Off     |
| `--no-plot` | Skip generating graph images | Off     |

#### Experiment 0 (Exp 0 Pipeline)

| Flag                    | Description                                                                                                  | Default               |
| ----------------------- | ------------------------------------------------------------------------------------------------------------ | --------------------- |
| `--run VARIANT`         | Which experiment to run: `baseline`, `move-ordering`, `transposition`, `baseline-vs-transposition`, or `all` | `baseline`            |
| `--games N`             | Total games to play                                                                                          | 50                    |
| `--depth N`             | Search depth for non-random agents                                                                           | 4                     |
| `--time-limit N`        | Max seconds per move                                                                                         | 5.0                   |
| `--plot-output-dir DIR` | Root output directory for graph images                                                                       | `results/exp0/images` |
| `--plot-label LABEL`    | Optional dataset label for graph legends                                                                     | (empty)               |

#### Experiments 1, 2, and 3

| Flag                                   | Description                                                     | Default   |
| -------------------------------------- | --------------------------------------------------------------- | --------- |
| `--games N`                            | Games per side per match-up/config/depth                        | 10        |
| `--depth N` (Exp 1-2 only)             | Search depth for non-random agents                              | 4         |
| `--depths D1 D2 ... (Exp 3 only)`      | Depths to evaluate                                              | `3 4 5 6` |
| `--time-limit N`                       | Max seconds per move                                            | 5.0       |
| `--seed N`                             | RNG seed for reproducible runs                                  | None      |
| `--stochastic-tiebreak`                | Randomly break near-equal best moves                            | Off       |
| `--opening-random-plies N`             | Random opening plies before normal search                       | 0         |
| `--plot-from-run-dir DIR` (Exp 3 only) | Regenerate graphs from existing result without re-running games | None      |

---

### Generating Charts from Existing Results

```bash
# Default (baseline vs random)
python -m experiments.plot_performance

# Custom datasets
python -m experiments.plot_performance \
  --dataset Baseline results/exp0/baseline_vs_random/exp0_baseline_vs_random.json results/exp0/baseline_vs_random/exp0_summary.json \
  --dataset "Move Ordering" results/exp0/move_ordering_vs_random/exp0_move_ordering_vs_random.json results/exp0/move_ordering_vs_random/exp0_move_ordering_random_summary.json
```

---

## Agents

| Agent             | Class                | Description                                           |
| ----------------- | -------------------- | ----------------------------------------------------- |
| **Baseline**      | `BaselineAgent`      | Minimax + Alpha-Beta, fixed depth, no move ordering   |
| **Move Ordering** | `MoveOrderingAgent`  | Baseline + Killer Heuristic + History Heuristic       |
| **Transposition** | `TranspositionAgent` | Move Ordering + Zobrist Hashing + Transposition Table |
| **Random**        | `RandomAgent`        | Uniform random legal move selection; control opponent |

### Evaluation Function

All search agents use the same heuristic:

```
h(s) = 3.0·M + 5.0·K + 0.1·L + 0.5·P + 0.3·C
```

| Feature       | Description                                                  |
| ------------- | ------------------------------------------------------------ |
| M — Material  | Piece count difference                                       |
| K — Kings     | King count difference (kings worth more than regular pieces) |
| L — Mobility  | Legal move count difference                                  |
| P — Promotion | Proximity of regular pieces to promotion rank                |
| C — Center    | Occupation of the eight central squares                      |

---

## Game Rules

This project implements **American Checkers** with the following rules:

- Red moves first
- Pieces move diagonally forward; kings move in all four diagonals
- **Forced capture rule**: if any capture is available, only captures are legal
- Multi-jump sequences count as a single turn
- A piece crowning mid-jump ends the turn immediately
- Draw conditions: threefold repetition, 50 consecutive non-capture moves, or 300-move hard cap

---

## Result File Structure

Experiments save JSON output to versioned subdirectories:

```
results/
└── exp0/
    └── baseline_vs_random/
        ├── d4_n50/
        │   ├── exp0_baseline_vs_random.json   # Per-game records
        │   └── exp0_summary.json              # Aggregate statistics
        │   └── images/                        # Charts
        └── exp0_baseline_vs_random.json       # Latest snapshot
```

---

## Notes

- The agent moving **second (BLACK)** may be at a slight disadvantage in early games. All experiments run each agent as both RED and BLACK with equal counts to balance this.
- Transposition table size is capped at **1,000,000 entries** with FIFO eviction.
- All experiment scripts are designed for submission to the **Purdue Gilbreth HPC cluster** via SLURM. Use `--seed` for reproducibility across nodes.
