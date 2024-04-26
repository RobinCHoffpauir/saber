from pybaseball import batting_stats, pitching_stats, team_batting
import pandas as pd
import numpy as np

# Get batting and pitching stats for a particular season
year = 2024
batting_data = batting_stats((year-1), year)
pitching_data = pitching_stats((year-1), year)

# Get team batting data for a particular season
team_batting_data = team_batting(year)

# Get starting pitchers for each team
# Replace these names with the actual starting pitchers for the game
home_team_starting_pitcher = input("Who is the home pitcher?: ")
away_team_starting_pitcher = input("WHo is the away pitcher?: ")

home_team_pitcher_data = pitching_data[pitching_data['Name']
                                       == home_team_starting_pitcher]
away_team_pitcher_data = pitching_data[pitching_data['Name']
                                       == away_team_starting_pitcher]


def simulate_game(home_team_batting, away_team_batting, home_pitcher, away_pitcher, num_simulations=1000):
    if home_pitcher.empty or away_pitcher.empty:
        print("One or both of the starting pitchers were not found in the dataset.")
        return None

    home_wins = 0
    away_wins = 0

    home_runs_per_game = home_team_batting['R'].values[0] / \
        home_team_batting['G'].values[0]
    away_runs_per_game = away_team_batting['R'].values[0] / \
        away_team_batting['G'].values[0]

    home_pitcher_runs_per_game = home_pitcher['R'].values[0] / \
        home_pitcher['G'].values[0]
    away_pitcher_runs_per_game = away_pitcher['R'].values[0] / \
        away_pitcher['G'].values[0]

    for _ in range(num_simulations):
        home_lambda = max(home_runs_per_game - away_pitcher_runs_per_game, 0)
        away_lambda = max(away_runs_per_game - home_pitcher_runs_per_game, 0)

        home_score = np.random.poisson(home_lambda)
        away_score = np.random.poisson(away_lambda)

        if home_score > away_score:
            home_wins += 1
        else:
            away_wins += 1

    return home_wins / num_simulations, away_wins / num_simulations


home_team = input("Who is the home team? ")
away_team = input("Who is the away team? ")

home_team_batting = team_batting_data[team_batting_data['Team'] == home_team]
away_team_batting = team_batting_data[team_batting_data['Team'] == away_team]

win_probabilities = simulate_game(
    home_team_batting, away_team_batting, home_team_pitcher_data, away_team_pitcher_data)
print(f"Win probabilities for {home_team} vs {away_team}: {win_probabilities}")
