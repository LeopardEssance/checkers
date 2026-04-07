"""
Experiment 1: Head-to-Head Agent Comparison

Research question: Do enhanced agents improve win rate and search
efficiency compared to baseline alternatives?

Match-ups (each played N games as RED + N games as BLACK):
  - Transposition vs Baseline
    - Transposition vs Move Ordering
  - Transposition vs Random
  - Baseline vs Random
    - Move Ordering vs Random

Metrics: win rate, average nodes/move, average time/move

Optional controls:
    - Seeded stochastic tie-breaks among near-equal best moves
    - Seeded random opening plies to diversify trajectories

Output:
  results/experiment1_head_to_head/d{depth}_n{games}/experiment1_head_to_head.json
  results/experiment1_head_to_head/d{depth}_n{games}/experiment1_summary.json
  results/experiment1_head_to_head/experiment1_head_to_head.json
  results/experiment1_head_to_head/experiment1_summary.json

Run: python -m experiments.experiment1_head_to_head
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass
import matplotlib.pyplot as plt

from src.agents.base import BaselineAgent, RandomAgent
from src.agents.move_ordering import MoveOrderingAgent
from src.agents.transposition import TranspositionAgent
from src.engine.board import BLACK, RED
from src.engine.game_runner import play_game


DEPTH = 4
NUM_GAMES = 10
TIME_LIMIT = 5.0
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

AGENT_DISPLAY = {
    "Baseline": "Baseline",
    "Move Ordering": "Move Ordering",
    "Transposition": "Transposition",
    "Random": "Random",
}

AGENT_COLORS = {
    "Transposition": "#B22222",  # red
    "Baseline": "#1f77b4",       # blue
    "Move Ordering": "#2E8B57",  # green
    "Random": "#696969",         # gray
}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return (sum((x - m) ** 2 for x in values) / (len(values) - 1)) ** 0.5


def _pct(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


@dataclass
class GameRecord:
    game_id: int
    matchup: str
    agent_a: str
    agent_b: str
    agent_a_side: str
    winner: str
    agent_a_won: bool
    draw: bool
    draw_reason: str
    move_limit_hit: bool
    total_moves: int
    agent_a_total_nodes: int
    agent_b_total_nodes: int
    agent_a_total_time_s: float
    agent_b_total_time_s: float
    agent_a_avg_nodes_per_move: float
    agent_a_avg_time_per_move_ms: float
    agent_b_avg_nodes_per_move: float
    agent_b_avg_time_per_move_ms: float
    agent_a_nodes_per_move: list[int]
    agent_a_time_per_move_ms: list[float]
    agent_b_nodes_per_move: list[int]
    agent_b_time_per_move_ms: list[float]
    avg_nodes_per_move: float
    avg_time_per_move_ms: float


@dataclass
class MatchSummary:
    matchup: str
    agent_a: str
    agent_b: str
    total_games: int
    wins: int
    losses: int
    draws: int
    win_rate_pct: float
    loss_rate_pct: float
    draw_rate_pct: float
    avg_nodes_per_move_mean: float
    avg_nodes_per_move_std: float
    avg_time_per_move_ms_mean: float
    avg_time_per_move_ms_std: float


@dataclass
class ExperimentSummary:
    total_matchups: int
    total_games: int
    depth: int
    games_per_side: int
    timestamp: str
    match_summaries: list[MatchSummary]


MATCHUPS = [
    ("Transposition vs Baseline", "Transposition", "Baseline"),
    ("Transposition vs Move Ordering", "Transposition", "Move Ordering"),
    ("Transposition vs Random", "Transposition", "Random"),
    ("Baseline vs Random", "Baseline", "Random"),
    ("Move Ordering vs Random", "Move Ordering", "Random"),
]


def _slug_matchup(text: str) -> str:
    return text.lower().replace(" ", "_")


def _display_agent(name: str) -> str:
    return AGENT_DISPLAY.get(name, name)


def _agent_color(name: str) -> str:
    return AGENT_COLORS.get(name, "#0072B2")


def _build_agent(name: str, color: str, depth: int, time_limit: float):
    if name == "Baseline":
        return BaselineAgent(color, depth=depth, time_limit=time_limit)
    if name == "Move Ordering":
        return MoveOrderingAgent(color, depth=depth, time_limit=time_limit)
    if name == "Transposition":
        return TranspositionAgent(color, depth=depth, time_limit=time_limit)
    if name == "Random":
        return RandomAgent(color)
    raise ValueError(f"Unsupported agent: {name}")


def _build_agent_with_seed(
    name: str,
    color: str,
    depth: int,
    time_limit: float,
    rng_seed: int | None,
    stochastic_tiebreak: bool,
):
    rng = random.Random(rng_seed) if rng_seed is not None else None

    if name == "Baseline":
        return BaselineAgent(
            color,
            depth=depth,
            time_limit=time_limit,
            rng=rng if stochastic_tiebreak else None,
        )
    if name == "Move Ordering":
        return MoveOrderingAgent(
            color,
            depth=depth,
            time_limit=time_limit,
            rng=rng if stochastic_tiebreak else None,
        )
    if name == "Transposition":
        return TranspositionAgent(
            color,
            depth=depth,
            time_limit=time_limit,
            rng=rng if stochastic_tiebreak else None,
        )
    if name == "Random":
        return RandomAgent(color, rng=rng)
    raise ValueError(f"Unsupported agent: {name}")


def _summarize_match(matchup: str, agent_a: str, agent_b: str, records: list[GameRecord]) -> MatchSummary:
    wins = sum(1 for r in records if r.agent_a_won)
    losses = sum(1 for r in records if not r.agent_a_won and not r.draw)
    draws = sum(1 for r in records if r.draw)
    nodes = [r.avg_nodes_per_move for r in records]
    times = [r.avg_time_per_move_ms for r in records]

    return MatchSummary(
        matchup=matchup,
        agent_a=agent_a,
        agent_b=agent_b,
        total_games=len(records),
        wins=wins,
        losses=losses,
        draws=draws,
        win_rate_pct=_pct(wins, len(records)),
        loss_rate_pct=_pct(losses, len(records)),
        draw_rate_pct=_pct(draws, len(records)),
        avg_nodes_per_move_mean=_mean(nodes),
        avg_nodes_per_move_std=_std(nodes),
        avg_time_per_move_ms_mean=_mean(times),
        avg_time_per_move_ms_std=_std(times),
    )


def run_experiment(
    num_games: int = NUM_GAMES,
    depth: int = DEPTH,
    time_limit: float = TIME_LIMIT,
    stochastic_tiebreak: bool = True,
    opening_random_plies: int = 0,
    seed: int | None = None,
    verbose: bool = False,
) -> tuple[list[GameRecord], ExperimentSummary]:
    print("=" * 60)
    print("Experiment 1: Head-to-Head Comparison")
    print(f"Depth={depth} | {num_games * 2} games per match-up")
    print(f"Stochastic tie-break: {'ON' if stochastic_tiebreak else 'OFF'}")
    print(f"Opening random plies: {opening_random_plies}")
    if seed is not None:
        print(f"Seed={seed}")
    print("=" * 60)

    all_records: list[GameRecord] = []
    match_summaries: list[MatchSummary] = []
    game_id = 0
    run_rng = random.Random(seed) if seed is not None else random.Random()

    for matchup, agent_a_name, agent_b_name in MATCHUPS:
        print(f"\n{matchup} ...")
        match_records: list[GameRecord] = []

        for color_a in [RED, BLACK]:
            color_b = BLACK if color_a == RED else RED
            for _ in range(num_games):
                game_seed = run_rng.randrange(2**32)
                agent_a = _build_agent_with_seed(
                    agent_a_name,
                    color_a,
                    depth,
                    time_limit,
                    rng_seed=game_seed ^ 0xA5A5A5A5,
                    stochastic_tiebreak=stochastic_tiebreak,
                )
                agent_b = _build_agent_with_seed(
                    agent_b_name,
                    color_b,
                    depth,
                    time_limit,
                    rng_seed=game_seed ^ 0x5A5A5A5A,
                    stochastic_tiebreak=stochastic_tiebreak,
                )
                opening_rng = random.Random(game_seed ^ 0xC3C3C3C3)

                if color_a == RED:
                    result = play_game(
                        agent_a,
                        agent_b,
                        tracked_player=RED,
                        opening_random_plies=opening_random_plies,
                        opening_rng=opening_rng,
                        verbose=verbose,
                    )
                else:
                    result = play_game(
                        agent_b,
                        agent_a,
                        tracked_player=BLACK,
                        opening_random_plies=opening_random_plies,
                        opening_rng=opening_rng,
                        verbose=verbose,
                    )

                rec = GameRecord(
                    game_id=game_id,
                    matchup=matchup,
                    agent_a=agent_a_name,
                    agent_b=agent_b_name,
                    agent_a_side=color_a,
                    winner=result.winner,
                    agent_a_won=(result.winner == color_a),
                    draw=result.draw,
                    draw_reason=result.draw_reason,
                    move_limit_hit=result.move_limit_hit,
                    total_moves=result.total_moves,
                    agent_a_total_nodes=(
                        sum(result.nodes_per_move_red)
                        if color_a == RED
                        else sum(result.nodes_per_move_black)
                    ),
                    agent_b_total_nodes=(
                        sum(result.nodes_per_move_black)
                        if color_a == RED
                        else sum(result.nodes_per_move_red)
                    ),
                    agent_a_total_time_s=(
                        sum(result.time_per_move_red_s)
                        if color_a == RED
                        else sum(result.time_per_move_black_s)
                    ),
                    agent_b_total_time_s=(
                        sum(result.time_per_move_black_s)
                        if color_a == RED
                        else sum(result.time_per_move_red_s)
                    ),
                    agent_a_avg_nodes_per_move=(
                        result.avg_nodes_per_move_red
                        if color_a == RED
                        else result.avg_nodes_per_move_black
                    ),
                    agent_a_avg_time_per_move_ms=(
                        result.avg_time_per_move_red_s * 1000.0
                        if color_a == RED
                        else result.avg_time_per_move_black_s * 1000.0
                    ),
                    agent_b_avg_nodes_per_move=(
                        result.avg_nodes_per_move_black
                        if color_a == RED
                        else result.avg_nodes_per_move_red
                    ),
                    agent_b_avg_time_per_move_ms=(
                        result.avg_time_per_move_black_s * 1000.0
                        if color_a == RED
                        else result.avg_time_per_move_red_s * 1000.0
                    ),
                    agent_a_nodes_per_move=(
                        result.nodes_per_move_red
                        if color_a == RED
                        else result.nodes_per_move_black
                    ),
                    agent_a_time_per_move_ms=[
                        t * 1000.0
                        for t in (
                            result.time_per_move_red_s
                            if color_a == RED
                            else result.time_per_move_black_s
                        )
                    ],
                    agent_b_nodes_per_move=(
                        result.nodes_per_move_black
                        if color_a == RED
                        else result.nodes_per_move_red
                    ),
                    agent_b_time_per_move_ms=[
                        t * 1000.0
                        for t in (
                            result.time_per_move_black_s
                            if color_a == RED
                            else result.time_per_move_red_s
                        )
                    ],
                    avg_nodes_per_move=(
                        result.avg_nodes_per_move_red
                        if color_a == RED
                        else result.avg_nodes_per_move_black
                    ),
                    avg_time_per_move_ms=(
                        result.avg_time_per_move_red_s * 1000.0
                        if color_a == RED
                        else result.avg_time_per_move_black_s * 1000.0
                    ),
                )
                all_records.append(rec)
                match_records.append(rec)
                game_id += 1

                status = "WIN" if rec.agent_a_won else ("DRAW" if rec.draw else "LOSS")
                side = "RED" if rec.agent_a_side == RED else "BLK"
                line = (
                    f"  Game {game_id:>3}  [{side}]  {status:4}  "
                    f"moves={rec.total_moves:>3}  "
                    f"{_display_agent(rec.agent_a)} nodes/move={rec.agent_a_avg_nodes_per_move:>8.1f}  "
                    f"ms/move={rec.agent_a_avg_time_per_move_ms:>7.2f}"
                )
                if rec.agent_b != "Random":
                    line += (
                        f"  |  {_display_agent(rec.agent_b)} nodes/move={rec.agent_b_avg_nodes_per_move:>8.1f}  "
                        f"ms/move={rec.agent_b_avg_time_per_move_ms:>7.2f}"
                    )
                print(line)

        summary = _summarize_match(matchup, agent_a_name, agent_b_name, match_records)
        match_summaries.append(summary)

        print(
            f"  {_display_agent(agent_a_name)}: W={summary.wins} L={summary.losses} D={summary.draws} "
            f"({summary.win_rate_pct:.1f}%) | "
            f"AvgNodes={summary.avg_nodes_per_move_mean:,.1f} | "
            f"AvgTime={summary.avg_time_per_move_ms_mean:.3f}ms"
        )

    experiment_summary = ExperimentSummary(
        total_matchups=len(MATCHUPS),
        total_games=len(all_records),
        depth=depth,
        games_per_side=num_games,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        match_summaries=match_summaries,
    )

    return all_records, experiment_summary


def save_results(
    records: list[GameRecord],
    summary: ExperimentSummary,
    results_dir: str = RESULTS_DIR,
) -> None:
    base_dir = os.path.join(results_dir, "experiment1_head_to_head")
    os.makedirs(base_dir, exist_ok=True)

    run_folder = f"d{summary.depth}_n{summary.games_per_side * 2}"
    run_dir = os.path.join(base_dir, run_folder)
    os.makedirs(run_dir, exist_ok=True)

    run_games_path = os.path.join(run_dir, "experiment1_head_to_head.json")
    run_summary_path = os.path.join(run_dir, "experiment1_summary.json")
    latest_games_path = os.path.join(base_dir, "experiment1_head_to_head.json")
    latest_summary_path = os.path.join(base_dir, "experiment1_summary.json")

    with open(run_games_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, indent=2)

    with open(run_summary_path, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)

    with open(latest_games_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, indent=2)

    with open(latest_summary_path, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)

    print("\nResults saved:")
    print(f"  {run_games_path}")
    print(f"  {run_summary_path}")
    print(f"  {latest_games_path}")
    print(f"  {latest_summary_path}")


def _save_graphs(
    records: list[GameRecord],
    summary: ExperimentSummary,
    results_dir: str = RESULTS_DIR,
) -> list[str]:
    base_dir = os.path.join(results_dir, "experiment1_head_to_head")
    run_folder = f"d{summary.depth}_n{summary.games_per_side * 2}"
    images_dir = os.path.join(base_dir, run_folder, "images")
    os.makedirs(images_dir, exist_ok=True)

    output_paths: list[str] = []

    for idx, match_summary in enumerate(summary.match_summaries):
        matchup = match_summary.matchup
        slug = _slug_matchup(matchup)
        matchup_records = [r for r in records if r.matchup == matchup]
        if not matchup_records:
            continue

        include_agent_b = match_summary.agent_b != "Random"

        # Outcomes chart for this matchup.
        outcomes_path = os.path.join(images_dir, f"{slug}_winrate_outcomes.png")
        fig, ax = plt.subplots(figsize=(8, 5))
        if include_agent_b:
            categories = ["Wins", "Losses", "Draws"]
            x = list(range(len(categories)))
            width = 0.35
            agent_a_vals = [match_summary.wins, match_summary.losses, match_summary.draws]
            agent_b_vals = [match_summary.losses, match_summary.wins, match_summary.draws]
            ax.bar(
                [i - width / 2 for i in x],
                agent_a_vals,
                width=width,
                label=_display_agent(match_summary.agent_a),
                color=_agent_color(match_summary.agent_a),
            )
            ax.bar(
                [i + width / 2 for i in x],
                agent_b_vals,
                width=width,
                label=_display_agent(match_summary.agent_b),
                color=_agent_color(match_summary.agent_b),
            )
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
        else:
            categories = ["Wins", "Losses", "Draws"]
            x = list(range(len(categories)))
            vals = [match_summary.wins, match_summary.losses, match_summary.draws]
            colors = ["#2E8B57", "#B22222", "#696969"]
            ax.bar(x, vals, color=colors)
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.legend([_display_agent(match_summary.agent_a)], frameon=False)

        ax.set_ylabel("Game Count")
        ax.set_title("Outcomes")
        ax.grid(axis="y", alpha=0.25)
        if include_agent_b:
            ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(outcomes_path, dpi=180)
        plt.close(fig)
        output_paths.append(outcomes_path)

        # Nodes chart for this matchup.
        nodes_path = os.path.join(images_dir, f"{slug}_avg_nodes_per_move.png")
        fig, ax = plt.subplots(figsize=(10, 6))
        nodes_a = [r.agent_a_avg_nodes_per_move for r in matchup_records]
        x_values = list(range(1, len(nodes_a) + 1))
        ax.plot(
            x_values,
            nodes_a,
            label=_display_agent(match_summary.agent_a),
            color=_agent_color(match_summary.agent_a),
            linewidth=1.8,
            alpha=0.9,
        )
        if include_agent_b:
            nodes_b = [r.agent_b_avg_nodes_per_move for r in matchup_records]
            ax.plot(
                x_values,
                nodes_b,
                label=_display_agent(match_summary.agent_b),
                color=_agent_color(match_summary.agent_b),
                linewidth=1.8,
                alpha=0.9,
            )
        ax.set_title("Nodes per Move by Game")
        ax.set_xlabel("Game #")
        ax.set_ylabel("Nodes / Move")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(nodes_path, dpi=180)
        plt.close(fig)
        output_paths.append(nodes_path)

        # Time chart for this matchup.
        time_path = os.path.join(images_dir, f"{slug}_avg_time_per_move.png")
        fig, ax = plt.subplots(figsize=(10, 6))
        time_a = [r.agent_a_avg_time_per_move_ms for r in matchup_records]
        ax.plot(
            x_values,
            time_a,
            label=_display_agent(match_summary.agent_a),
            color=_agent_color(match_summary.agent_a),
            linewidth=1.8,
            alpha=0.9,
        )
        if include_agent_b:
            time_b = [r.agent_b_avg_time_per_move_ms for r in matchup_records]
            ax.plot(
                x_values,
                time_b,
                label=_display_agent(match_summary.agent_b),
                color=_agent_color(match_summary.agent_b),
                linewidth=1.8,
                alpha=0.9,
            )
        ax.set_title("Time per Move by Game")
        ax.set_xlabel("Game #")
        ax.set_ylabel("Milliseconds / Move")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(time_path, dpi=180)
        plt.close(fig)
        output_paths.append(time_path)

    return output_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 1: Head-to-Head Comparison")
    parser.add_argument("--games", type=int, default=NUM_GAMES, help="Games per side per match-up")
    parser.add_argument("--depth", type=int, default=DEPTH, help="Search depth for non-random agents")
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT, help="Time limit per move (s)")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducible stochastic runs")
    parser.add_argument(
        "--stochastic-tiebreak",
        action="store_true",
        help="Randomly break near-equal best-move ties in search agents",
    )
    parser.add_argument(
        "--opening-random-plies",
        type=int,
        default=0,
        help="Number of opening plies chosen randomly before normal search",
    )
    parser.add_argument("--verbose", action="store_true", help="Print board each move")
    parser.add_argument("--no-save", action="store_true", help="Skip writing results to disk")
    parser.add_argument("--no-plot", action="store_true", help="Skip graph generation")
    args = parser.parse_args()

    records, summary = run_experiment(
        num_games=args.games,
        depth=args.depth,
        time_limit=args.time_limit,
        stochastic_tiebreak=args.stochastic_tiebreak,
        opening_random_plies=max(0, args.opening_random_plies),
        seed=args.seed,
        verbose=args.verbose,
    )

    if not args.no_save:
        save_results(records, summary)

    if args.no_plot:
        raise SystemExit(0)

    output_paths = _save_graphs(records, summary)
    print("\nSaved graphs:")
    for path in output_paths:
        print(f"  {path}")
