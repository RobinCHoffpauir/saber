# -*- coding: utf-8 -*-


import pandas as pd
import pybaseball as pyb
from pybaseball import schedule_and_record
from collections import defaultdict
import sqlite3

pyb.cache.enable()

# Function to calculate the expected outcome of a match


def expected_outcome(rating1, rating2):
    return 1 / (1 + 10 ** ((rating2 - rating1) / 400))


# Function to update the ELO rating


def update_elo(winner_elo, loser_elo, k_factor):
    expected_win = expected_outcome(winner_elo, loser_elo)
    change_in_elo = k_factor * (1 - expected_win)
    winner_elo += change_in_elo
    loser_elo -= change_in_elo
    return winner_elo, loser_elo


# Get the MLB schedule and record for the current year
for yr in range(2000,2023):
    year = yr
    elo_dict = defaultdict(lambda: 1500)  # Teams start with a ELO of 1500

    # Go through each team
    for team in [
        "LAA",
        "HOU",
        "OAK",
        "TOR",
        "ATL",
        "MIL",
        "STL",
        "CHC",
        "ARI",
        "LAD",
        "SFG",
        "CLE",
        "SEA",
        "MIA",
        "NYM",
        "WSN",
        "BAL",
        "SDP",
        "PHI",
        "PIT",
        "TEX",
        "TBR",
        "BOS",
        "CIN",
        "COL",
        "KCR",
        "DET",
        "MIN",
        "CHW",
        "NYY",
    ]:
        retro = pyb.team_results
        sched = retro.schedule_and_record(year, team)
        df = pd.DataFrame(sched)
        df["win_loss"] = df["W/L"]
        del df["W/L"]
        # Go through each game in the schedule
        for i, row in enumerate(df.itertuples()):
            if pd.isnull(row.win_loss):
                continue
            # Ignore games that haven't been played yet
            if row.win_loss == "W" or "W-wo":
                winner, loser = row.Opp, team
            else:
                winner, loser = team, row.Opp
            # Update the ELO ratings of the teams
            # Increase the k-factor for the most recent 15 games
            k_factor = 30 if i >= len(df) - 15 else 20
            # Update the ELO ratings of the teams
            elo_dict[winner], elo_dict[loser] = update_elo(
                elo_dict[winner], elo_dict[loser], 50
            )  # 20 is the k-factor

        df = pd.DataFrame(list(elo_dict.items()), columns=["Team", "ELO"])
        df.to_csv(f'../data/elo/{year}_elo.csv')