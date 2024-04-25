import requests
from datetime import datetime, timedelta
import pandas as pd
import json 
import odds, utils, scores
def extract_scores(row):
    scores = row['scores']  # Assume it's already a list of dictionaries
    home_score = None
    away_score = None
    for score in scores:
        if score['name'] == row['home_team']:
            home_score = score['score']
        elif score['name'] == row['away_team']:
            away_score = score['score']
    return pd.Series([home_score, away_score])
                     
                     
# Calculate yesterday's date for output or logging, not for API request
today_date = datetime.today() 
yesterday_date = today_date - timedelta(days=1)
yesterday_str = yesterday_date.strftime('%Y-%m-%d')

# API settings
API_KEY = '1628e7b26fd53acf9eef611ac3466368'
SPORT = 'baseball_mlb'
DATE_FORMAT = 'iso'

# Build the URL for the API request to the scores endpoint
url = f"https://api.the-odds-api.com/v4/sports/{SPORT}/scores/"
params = {
    'apiKey': API_KEY,
    'daysFrom': 2,  # Let's try with '1', which should be valid
    'dateFormat': DATE_FORMAT
}

def retrieve_scores():
    # Make the API request
    response = requests.get(url, params=params)

    # Check if the request was successful
    if response.status_code == 200:

        games = response.json()
        games_df = pd.DataFrame(games)
        preview = pd.read_csv(f'data/preview/{yesterday_str}_preview.csv')
        # Filter the DataFrame to only include rows where 'completed' is True
        completed_games_df = games_df[games_df['completed'] == True]

        # Then perform operations on this filtered DataFrame
        if not completed_games_df.empty:
            completed_games_df[['home_score', 'away_score']] = completed_games_df.apply(extract_scores, axis=1)

    else:
        print(f"Failed to retrieve data: {response.status_code}, {response.text}")

    # Print remaining API usage if available
    if 'x-requests-remaining' in response.headers:
        print(f"Requests remaining: {response.headers['x-requests-remaining']}")
    if 'x-quota-remaining' in response.headers:
        print(f"Quota remaining: {response.headers['x-quota-remaining']}")

    completed_games_df = completed_games_df.drop(['completed','sport_key','sport_title','last_update','scores'],axis=1)
    completed_games_df.to_csv(f'data/scores/{yesterday_str}_scores.csv')
    print('Completed games and scores saved to data/scores/')
    
def main():
    retrieve_scores()
    
if __name__ == '__main__':
    main()
    
    