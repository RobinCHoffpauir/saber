# -*- coding: utf-8 -*-
import pandas as pd
from pybaseball import team_batting, team_pitching
from pybaseball.lahman import batting
import pybaseball as pyb
pyb.cache.enable()
import sqlite3

teams = ['BOS','NYY','TBR','KCR','CHW','BAL','CLE','MIN','DET','HOU','LAA',
         'SEA','TEX','OAK','WSN','MIA','ATL','NYM','PHI','CHC','MIL','STL',
         'PIT','CIN','LAD','ARI','COL','SDP','SFG','TOR']
divisions = {
    "AL East": {"BOS", "NYY", "TBR", "TOR", "BAL"},
    "AL Central": {"CLE", "DET", "CHW", "KCR", "MIN"},
    "AL West": {"LAA", "HOU", "SEA", "TEX", "OAK"},
    "NL East": {"WSN", "NYM", "ATL", "MIA", "PHI"},
    "NL Central": {"CHC", "MIL", "STL", "PIT", "CIN"},
    "NL West": {"LAD", "ARI", "COL", "SDP", "SFG"}
}

def create_winning_percentage(year):
    # Create the team_dfs dict needed to run script
    year = year
    team_batting_data = team_batting(year) #fetch team_batting data
    team_pitching_data = team_pitching(year)  #fetch team_pitching data
    team_dfs = {}  # Create an empty #dictionary to store DataFrames #for each team
    db_path = f"Betting/data/databases/{year}_schedule_record.db"
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    team_dfs = {}
    ############################
    for div in divisions:
        for team in divisions.get(div):
            c.execute(f"SELECT * FROM {team}")
            rows = c.fetchall()
            columns = [column[0] for column in c.description]
            team_dfs[team] = pd.DataFrame(rows, columns=columns)


    # Merge the dataframes
    team_data = pd.merge(team_batting_data[['Team', 'PA', 'R', 'wRC']], team_pitching_data[[
                         'Team', 'ER', 'IP', 'FIP']], on='Team')
    runs, runs_allowed = team_batting_data[[
        'Team', 'R']], team_pitching_data[['Team', 'ER']]
    run_data = pd.merge(runs, runs_allowed, on='Team')
    # Calculate expected runs scored and allowed
    team_data['Expected Runs Scored'] = team_data['wRC'] / team_data['IP']*9
    team_data['Expected Runs Allowed'] = team_data['FIP']

    # Calculate expected winning percentage
    team_data['Created Winning %'] = team_data['Expected Runs Scored']**2 / \
        (team_data['Expected Runs Scored']**2 +
         team_data['Expected Runs Allowed']**2)
    run_data['Pythag Expected %'] = run_data['R']**2 / \
        (run_data['R']**2 + run_data['ER']**2)
    compare = pd.merge(run_data
    [['Team', 'Pythag Expected %']], team_data[[
                       'Team', 'Created Winning %']], on='Team')
    compare['Tm'] = compare['Team']
    compare = compare.drop('Team', axis=1)
    for x, df in team_dfs.items():
        df['Outcome'] = df['W/L'].apply(lambda x: 1 if x == 'W' or x == 'W-W/O' else 0)
        # Step 1: Split the 'W-L' column into two separate columns
        df[['Wins', 'Losses']] = df['W-L'].str.split('-', expand=True)

        # Step 2: Convert these columns to numeric
        df['Wins'] = pd.to_numeric(df['Wins'])
        df['Losses'] = pd.to_numeric(df['Losses'])

        # Step 3: Calculate the Winning Percentage
        df['Winning Percentage'] = df['Wins'] / (df['Wins'] + df['Losses'])
    final_win_percentages = {team: df['Winning Percentage'].iloc[-1] for team, df in team_dfs.items()}

    final_win_perc_df = pd.DataFrame(list(final_win_percentages.items()), 
                                     columns=['Tm', 'Final Winning Percentage']) 
    compare = pd.merge(compare, final_win_perc_df, on='Tm', how='left')
    compare.index= compare['Tm']
    compare = compare.drop('Tm', axis=1)
    compare['Pythag Difference'] = compare['Final Winning Percentage'] - compare['Pythag Expected %']
    compare['Composite Difference'] = compare['Final Winning Percentage'] - compare['Created Winning %']
    print(compare)
