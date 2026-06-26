"""Elo feature calculations for processed game data."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd


START_ELO = 1500.0
ELO_K = 20.0
ELO_HOME_ADVANTAGE = 65.0
ELO_CARRYOVER = 0.75


def as_int(value: Any) -> int:
    return int(value)


def expected_home_win_probability(home_elo: float, away_elo: float) -> float:
    rating_gap = (home_elo + ELO_HOME_ADVANTAGE) - away_elo
    return 1 / (1 + 10 ** (-rating_gap / 400))


def updated_elos(home_elo: float, away_elo: float, home_win: int) -> tuple[float, float]:
    expected_home_win = expected_home_win_probability(home_elo, away_elo)
    delta = ELO_K * (home_win - expected_home_win)
    return home_elo + delta, away_elo - delta


def final_flat_elos(games: pd.DataFrame) -> dict[int, float]:
    elos: defaultdict[int, float] = defaultdict(lambda: START_ELO)
    for game in games.itertuples(index=False):
        home_team_id = as_int(game.HOME_TEAM_ID)
        away_team_id = as_int(game.AWAY_TEAM_ID)
        elos[home_team_id], elos[away_team_id] = updated_elos(
            elos[home_team_id],
            elos[away_team_id],
            as_int(game.HOME_WIN),
        )
    return dict(elos)


def carryover_elos(previous_games: pd.DataFrame | None) -> dict[int, float]:
    if previous_games is None:
        return {}

    return {
        team_id: START_ELO + (ELO_CARRYOVER * (elo - START_ELO))
        for team_id, elo in final_flat_elos(previous_games).items()
    }


def build_elo_features(
    games: pd.DataFrame,
    previous_games: pd.DataFrame | None,
) -> pd.DataFrame:
    flat_elos: defaultdict[int, float] = defaultdict(lambda: START_ELO)
    carry_elos: defaultdict[int, float] = defaultdict(lambda: START_ELO)
    carry_elos.update(carryover_elos(previous_games))

    sos_sums: defaultdict[int, float] = defaultdict(float)
    sos_counts: defaultdict[int, int] = defaultdict(int)
    rows = []
    for game in games.sort_values(["GAME_DATE", "GAME_ID"]).itertuples(index=False):
        home_team_id = as_int(game.HOME_TEAM_ID)
        away_team_id = as_int(game.AWAY_TEAM_ID)
        home_flat_elo = flat_elos[home_team_id]
        away_flat_elo = flat_elos[away_team_id]
        home_carry_elo = carry_elos[home_team_id]
        away_carry_elo = carry_elos[away_team_id]
        home_sos = (
            sos_sums[home_team_id] / sos_counts[home_team_id]
            if sos_counts[home_team_id]
            else float("nan")
        )
        away_sos = (
            sos_sums[away_team_id] / sos_counts[away_team_id]
            if sos_counts[away_team_id]
            else float("nan")
        )

        rows.extend(
            [
                {
                    "GAME_ID": game.GAME_ID,
                    "TEAM_ID": home_team_id,
                    "ELO_FLAT_PRE_GAME": home_flat_elo,
                    "ELO_CARRYOVER_PRE_GAME": home_carry_elo,
                    "STRENGTH_OF_SCHEDULE_ELO_TO_DATE": home_sos,
                },
                {
                    "GAME_ID": game.GAME_ID,
                    "TEAM_ID": away_team_id,
                    "ELO_FLAT_PRE_GAME": away_flat_elo,
                    "ELO_CARRYOVER_PRE_GAME": away_carry_elo,
                    "STRENGTH_OF_SCHEDULE_ELO_TO_DATE": away_sos,
                },
            ]
        )

        sos_sums[home_team_id] += away_flat_elo
        sos_counts[home_team_id] += 1
        sos_sums[away_team_id] += home_flat_elo
        sos_counts[away_team_id] += 1

        flat_elos[home_team_id], flat_elos[away_team_id] = updated_elos(
            home_flat_elo,
            away_flat_elo,
            as_int(game.HOME_WIN),
        )
        carry_elos[home_team_id], carry_elos[away_team_id] = updated_elos(
            home_carry_elo,
            away_carry_elo,
            as_int(game.HOME_WIN),
        )

    return pd.DataFrame(rows)
