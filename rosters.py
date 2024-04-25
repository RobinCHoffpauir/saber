import requests, pandas as pd, sqlite3

import sqlalchemy as sa

# create and connect to the database
engine = sa.create_engine('sqlite:///Betting/data/rosters/mlb_rosters.db')
conn = engine.connect()
cursor = conn.connection.cursor()

# create list of team names
teams = ['BOS', 'NYY', 'TB', 'KC', 'CHW', 'BAL', 'CLE', 'MIN', 
         'DET', 'HOU', 'LAA', 'SEA', 'TEX', 'OAK', 'WAS', 'MIA', 
         'ATL', 'NYM', 'PHI', 'CHC', 'MIL', 'STL', 'PIT', 'CIN', 
         'LAD', 'ARI', 'COL', 'SD', 'SF', 'TOR']

url = "https://tank01-mlb-live-in-game-real-time-statistics.p.rapidapi.com/getMLBTeamRoster" 
for team in teams:
    querystring = {"teamAbv":{team},"getStats":"false"} 
    headers = { "X-RapidAPI-Key": "b501e0bf75mshf881260fcf61406p1a7f13jsnbce859978dd1", 
               "X-RapidAPI-Host": "tank01-mlb-live-in-game-real-time-statistics.p.rapidapi.com" } 
    response = requests.get(url, headers=headers, params=querystring)
    data = response.json()

    if 'body' in data and 'roster' in data['body']:
        roster_df = pd.json_normalize(data['body']['roster'])

        # Define the desired columns and check if they exist in the DataFrame
        desired_columns = [
            'longName', 'team', 'teamAbv', 'pos', 'bat', 'throw', 'height', 'weight',
            'bDay', 'college', 'espnLink', 'mlbLink', 
            'stats.Hitting.avg', 'stats.Hitting.HR',
            'stats.Pitching.ERA', 'stats.Fielding.E'
        ]
        available_columns = [col for col in desired_columns if col in roster_df.columns]

        # Select and rename columns as needed, based on available data
        roster_df = roster_df[available_columns]
        column_renames = {
            'stats.Hitting.avg': 'Batting Average',
            'stats.Hitting.HR': 'Home Runs',
            'stats.Pitching.ERA': 'ERA',
            'stats.Fielding.E': 'Errors'
        }
        rename_columns = {col: column_renames[col] for col in available_columns if col in column_renames}
        roster_df.rename(columns=rename_columns, inplace=True)
        roster_df.fillna('None',axis=1, inplace=True)
        roster_df.to_sql(name=team, con=conn, if_exists='replace', index=False)
    else:
        print(f"for {team} No roster data available.")

















# df = pd.read_json(response.text)
# col_roster = df.to_dict()
# data = {}
# for col, row in col_roster.items():
#     data[col] = row
