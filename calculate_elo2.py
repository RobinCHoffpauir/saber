import pandas as pd
import pybaseball as pyb
from collections import defaultdict
import os
from Betting.utils import *
setup_logging()


pyb.cache.enable()

id_changes = {
    'ANA': 'LAA',
    'TBD': 'TBR',
    # Continue to add any further changes here
}

# Main processing loop
for year in range(2000, 2024):
    current_elo_ratings = load_elo_ratings(year - 1)
    
    # Apply ID changes for the new season
    new_elo_ratings = defaultdict(lambda: 1500)  # Start with a default dictionary
    for team, elo in current_elo_ratings.items():
        new_team_id = id_changes.get(team, team)  # Get the new team ID or use the current one
        new_elo_ratings[new_team_id] = elo

    current_elo_ratings = new_elo_ratings

    for team in pyb.team_ids(year)['teamIDBR']:
        sched = pyb.team_results.schedule_and_record(year, team)
        df = pd.DataFrame(sched)
        df["win_loss"] = df["W/L"]
        del df["W/L"]

        for row in df.itertuples():
            if row.win_loss in ["W", "W-wo"]:
                winner, loser = row.Opp, team
            elif row.win_loss in ["L", "L-wo"]:
                winner, loser = team, row.Opp
            else:
                continue  # Skip non-decisive games

            # Safely get ELO ratings with defaults
            winner_elo = get_elo(winner, current_elo_ratings)
            loser_elo = get_elo(loser, current_elo_ratings)

            # Update ELO ratings
            k_factor = 30 if 'Recent' in str(row.Date) else 20
            updated_winner_elo, updated_loser_elo = update_elo(winner_elo, loser_elo, k_factor)
            current_elo_ratings[winner] = updated_winner_elo
            current_elo_ratings[loser] = updated_loser_elo

    # Save ELO ratings for the current year
    save_elo_ratings(current_elo_ratings, year)

print("ELO ratings processing completed for all years.")