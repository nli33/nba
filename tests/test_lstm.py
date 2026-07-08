from __future__ import annotations

import numpy as np
import pandas as pd

from nba_predictor.models import lstm


FEATURES = ["NET", "IS_HOME"]

# Four teams with fixed strengths; the stronger team always wins.
STRENGTHS = {1: 9.0, 2: 3.0, 3: -3.0, 4: -9.0}


def build_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    team_rows: list[dict[str, object]] = []
    game_rows: list[dict[str, object]] = []
    game_index = 0
    for game_round in range(12):
        for home, away in ((1, 2), (3, 4), (1, 3), (2, 4), (1, 4), (2, 3)):
            game_id = f"G{game_index:03d}"
            date = pd.Timestamp("2025-01-01") + pd.Timedelta(days=game_index)
            home_win = int(STRENGTHS[home] > STRENGTHS[away])
            for team_id, is_home in ((home, 1.0), (away, 0.0)):
                team_rows.append(
                    {
                        "GAME_ID": game_id,
                        "GAME_DATE": date,
                        "TEAM_ID": team_id,
                        "NET": STRENGTHS[team_id],
                        "IS_HOME": is_home,
                    }
                )
            game_rows.append(
                {
                    "GAME_ID": game_id,
                    "GAME_DATE": date,
                    "SEASON": "2024-25",
                    "HOME_TEAM_ID": home,
                    "HOME_TEAM_ABBREVIATION": f"H{home}",
                    "AWAY_TEAM_ID": away,
                    "AWAY_TEAM_ABBREVIATION": f"A{away}",
                    "HOME_WIN": home_win,
                }
            )
            game_index += 1
    return pd.DataFrame(game_rows), pd.DataFrame(team_rows)


def test_prior_window_uses_only_games_before_target() -> None:
    _, team_features = build_frames()
    store = lstm.build_sequence_store(team_features, FEATURES)

    team_games = (
        team_features[team_features["TEAM_ID"] == 1]
        .sort_values(["GAME_DATE", "GAME_ID"])["GAME_ID"]
        .tolist()
    )
    target = team_games[4]
    window = store.prior_window(1, target, sequence_length=10)

    assert window is not None
    # Exactly the four games that precede the target, none from the target onward.
    assert len(window) == 4


def test_first_game_has_no_prior_history() -> None:
    _, team_features = build_frames()
    store = lstm.build_sequence_store(team_features, FEATURES)
    first_game = (
        team_features[team_features["TEAM_ID"] == 1]
        .sort_values(["GAME_DATE", "GAME_ID"])["GAME_ID"]
        .iloc[0]
    )
    window = store.prior_window(1, first_game, sequence_length=10)
    assert window is not None
    assert len(window) == 0


def test_padded_window_is_front_padded_to_length() -> None:
    scaler = lstm.SequenceScaler(mean=np.zeros(2), std=np.ones(2))
    window = np.array([[1.0, 1.0], [2.0, 0.0]])
    padded = scaler.padded(window, sequence_length=4)
    assert padded.shape == (4, 2)
    # Real rows land at the end; padding zeros at the front.
    assert np.array_equal(padded[:2], np.zeros((2, 2)))
    assert np.array_equal(padded[2:], window)


def test_train_and_evaluate_learns_stronger_team() -> None:
    model_games, team_features = build_frames()
    artifact = lstm.train_lstm_from_frames(
        model_games,
        team_features,
        "2024-25",
        FEATURES,
        sequence_length=5,
        min_history=3,
        hidden_size=16,
        epochs=40,
    )
    assert artifact.games_trained > 0

    evaluation = lstm.evaluate_lstm_on_frames(
        artifact, model_games, team_features, "2024-25", ["2024-25"]
    )
    # The outcome is fully determined by team strength; the model should learn it.
    assert evaluation.accuracy_when_predicted > 0.9
