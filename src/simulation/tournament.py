"""
Group stage + knockout bracket simulator for 48-team / 12-group FIFA World Cup format.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Optional

from src.simulation.monte_carlo import MatchSimulator

logger = logging.getLogger(__name__)

TOURNAMENT = "FIFA World Cup"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TeamStanding:
    team: str
    group: str
    rank: int = 0          # 1st, 2nd, 3rd, 4th within group
    points: int = 0
    gf: int = 0            # goals for
    ga: int = 0            # goals against

    @property
    def gd(self) -> int:
        return self.gf - self.ga

    def sort_key(self) -> tuple:
        return (-self.points, -self.gd, -self.gf)


STAGES = ["group", "r32", "r16", "qf", "sf", "final", "champion"]


# ---------------------------------------------------------------------------
# Group stage
# ---------------------------------------------------------------------------

def simulate_group(
    group_id: str,
    teams: list[str],
    sim: MatchSimulator,
) -> list[TeamStanding]:
    """Round-robin within group. Returns standings sorted by points → GD → GF."""
    standings = {t: TeamStanding(team=t, group=group_id) for t in teams}

    for i, home in enumerate(teams):
        for away in teams[i + 1:]:
            hg, ag = sim.simulate_match(home, away, TOURNAMENT, neutral=False)
            standings[home].gf += hg
            standings[home].ga += ag
            standings[away].gf += ag
            standings[away].ga += hg
            if hg > ag:
                standings[home].points += 3
            elif hg == ag:
                standings[home].points += 1
                standings[away].points += 1
            else:
                standings[away].points += 3

    ranked = sorted(standings.values(), key=lambda s: s.sort_key())
    for i, s in enumerate(ranked):
        s.rank = i + 1
    return ranked


def simulate_all_groups(
    groups: dict[str, list[str]],
    sim: MatchSimulator,
) -> dict[str, list[TeamStanding]]:
    results = {}
    for gid, teams in groups.items():
        results[gid] = simulate_group(gid, teams, sim)
    return results


# ---------------------------------------------------------------------------
# 3rd-place selection (best 8 of 12 third-place teams)
# ---------------------------------------------------------------------------

def select_best_thirds(
    group_results: dict[str, list[TeamStanding]],
) -> list[TeamStanding]:
    thirds = [standings[2] for standings in group_results.values()]  # rank=3 → index 2
    thirds_sorted = sorted(thirds, key=lambda s: s.sort_key())
    return thirds_sorted[:8]


# ---------------------------------------------------------------------------
# Knockout bracket
# ---------------------------------------------------------------------------

def _simulate_knockout_match(
    team_a: str,
    team_b: str,
    sim: MatchSimulator,
    neutral: bool = True,
) -> str:
    """Simulate a knockout match. Penalties handled as 50/50 if draw after 90 min."""
    hg, ag = sim.simulate_match(team_a, team_b, TOURNAMENT, neutral=neutral)
    if hg > ag:
        return team_a
    if ag > hg:
        return team_b
    # Draw → penalty shootout (50/50)
    return random.choice([team_a, team_b])




def _run_bracket(
    pairs: list[tuple[str, str]],
    sim: MatchSimulator,
) -> dict[str, str]:
    """
    Single-pass bracket. Returns {team: stage_reached}.
    Strategy: set winners to NEXT stage so losers naturally retain current stage.
    Init all teams to "r32" (eliminated in R32 by default).
    R32 winners → "r16", R16 winners → "qf", QF → "sf", SF → "final",
    Final winner → "champion", final loser keeps "final".
    """
    stage_reached: dict[str, str] = {}
    for a, b in pairs:
        stage_reached[a] = "r32"
        stage_reached[b] = "r32"

    # (current_stage, next_stage_for_winners)
    round_map = [("r32", "r16"), ("r16", "qf"), ("qf", "sf"), ("sf", "final")]
    current = list(pairs)

    for _stage, next_stage in round_map:
        survivors: list[str] = []
        for a, b in current:
            winner = _simulate_knockout_match(a, b, sim)
            stage_reached[winner] = next_stage  # advance winner; loser keeps current stage
            survivors.append(winner)
        current = list(zip(survivors[::2], survivors[1::2]))

    # Final: 1 pair of teams both labelled "final"
    for a, b in current:
        winner = _simulate_knockout_match(a, b, sim)
        loser = b if winner == a else a
        stage_reached[winner] = "champion"
        stage_reached[loser] = "final"  # runner-up; already "final" but explicit

    return stage_reached


# ---------------------------------------------------------------------------
# Full tournament simulation (single run)
# ---------------------------------------------------------------------------

def simulate_tournament_once(
    groups: dict[str, list[str]],
    sim: MatchSimulator,
) -> dict[str, str]:
    """Run one complete tournament. Returns {team: stage_reached}."""
    all_teams = [t for teams in groups.values() for t in teams]
    stage_reached = {t: "group_exit" for t in all_teams}

    # Group stage
    group_results = simulate_all_groups(groups, sim)

    # Collect qualified teams
    qualified: list[TeamStanding] = []
    for standings in group_results.values():
        qualified.extend(standings[:2])  # top 2 from each group

    best_thirds = select_best_thirds(group_results)
    qualified.extend(best_thirds)  # 8 best 3rd-place teams → total 32

    for s in qualified:
        stage_reached[s.team] = "qualified"

    # Build R32 pairs
    winners = sorted([s for s in qualified if s.rank == 1], key=lambda s: s.group)
    runners = sorted([s for s in qualified if s.rank == 2], key=lambda s: s.group)
    thirds = sorted([s for s in best_thirds], key=lambda s: s.sort_key())

    seeded = winners + runners + thirds
    bracket: list[str] = [s.team for s in seeded]
    group_of = {s.team: s.group for s in qualified}

    pairs: list[tuple[str, str]] = []
    for i in range(16):
        a = bracket[i]
        b = bracket[31 - i]
        if group_of.get(a) == group_of.get(b):
            for offset in [1, -1, 2, -2, 3, -3]:
                j = 31 - i + offset
                if 16 <= j < 32 and group_of.get(bracket[j]) != group_of.get(a):
                    bracket[31 - i], bracket[j] = bracket[j], bracket[31 - i]
                    b = bracket[31 - i]
                    break
        pairs.append((a, b))

    # Knockout
    knockout_results = _run_bracket(pairs, sim)
    stage_reached.update(knockout_results)

    return stage_reached


# ---------------------------------------------------------------------------
# Monte Carlo aggregation
# ---------------------------------------------------------------------------

def aggregate_simulations(
    all_runs: list[dict[str, str]],
) -> dict[str, dict[str, float]]:
    """
    Aggregate N simulation runs into probability tables.
    Returns {team: {champion_prob, final_prob, sf_prob, qf_prob, r16_prob, r32_prob, group_exit_prob}}.
    """
    n = len(all_runs)
    if n == 0:
        return {}

    all_teams = set(k for run in all_runs for k in run)
    stage_order = ["champion", "final", "sf", "qf", "r16", "r32", "qualified", "group_exit"]

    results: dict[str, dict[str, float]] = {}
    for team in sorted(all_teams):
        counts: dict[str, int] = {s: 0 for s in stage_order}
        for run in all_runs:
            s = run.get(team, "group_exit")
            if s in counts:
                counts[s] += 1
        results[team] = {k: round(v / n, 4) for k, v in counts.items()}

    return results
