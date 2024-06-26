#%% Import required libraries
from bs4 import BeautifulSoup

import requests
import pandas as pd
import covers
from helpers import *
from datetime import datetime

team_names = {
    'BOS': 'Boston Red Sox',
    'NYY': 'New York Yankees',
    'TBR': 'Tampa Bay Rays',
    'KCR': 'Kansas City Royals',
    'CHW': 'Chicago White Sox',
    'BAL': 'Baltimore Orioles',
    'CLE': 'Cleveland Guardians',  # Updated from Indians
    'MIN': 'Minnesota Twins',
    'DET': 'Detroit Tigers',
    'HOU': 'Houston Astros',
    'LAA': 'Los Angeles Angels',
    'SEA': 'Seattle Mariners',
    'TEX': 'Texas Rangers',
    'OAK': 'Oakland Athletics',
    'WSN': 'Washington Nationals',
    'MIA': 'Miami Marlins',
    'ATL': 'Atlanta Braves',
    'NYM': 'New York Mets',
    'PHI': 'Philadelphia Phillies',
    'CHC': 'Chicago Cubs',
    'MIL': 'Milwaukee Brewers',
    'STL': 'St. Louis Cardinals',
    'PIT': 'Pittsburgh Pirates',
    'CIN': 'Cincinnati Reds',
    'LAD': 'Los Angeles Dodgers',
    'ARI': 'Arizona Diamondbacks',
    'COL': 'Colorado Rockies',
    'SDP': 'San Diego Padres',
    'SFG': 'San Francisco Giants',
    'TOR': 'Toronto Blue Jays'
}


def get_covers():
    # The URL to scrape
    url = 'https://www.covers.com/sport/baseball/mlb/printsheet'
    
    today_date = datetime.today() 
    today_str = today_date.strftime('%Y-%m-%d')
    # Send an HTTP request to the URL
    response = requests.get(url)

      # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Initialize a list to store all rows data
    table_data = []

    # Find all 'tr' elements with the class 'highlight' or 'odd-rows'
    rows = soup.find_all('tr', class_=['highlight', 'odd-rows'])

    # Iterate over each row
    for row in rows:
        # For each 'tr' element, find all 'td' elements and extract the text
        cols = row.find_all('td')
        cols_text = [col.text.strip() for col in cols]
        table_data.append(cols_text)

    # Define column names, assuming these are known and match the data provided
    columns = [
        'Game Number', 'Team', 'Pitcher', 'Rest', 'Season WL', 'Season ERA', 
        'Season WHIP', 'Last 3 WL', 'Last 3 IP', 'Last 3 ERA', 'Last 3 WHIP', 
        'K', 'HR', 'Team WLCS', 'Streak WL', 'Streak O/U'
    ]

    # Create a dataframe importing the CSV file from ../data/elo/2024_elo.csv
    elo = pd.read_csv(f'../../data/elo/2024_elo.csv')

    # Create a DataFrame using table_data
    df = pd.DataFrame(table_data, columns=columns)
    df = pd.merge(df, elo, on='Team')
    
    
    df['outcome_name'] = df['Team'].map(team_names)
    df = df.drop(['Game Number','Team'],axis=1)
    df.to_csv(f'../../data/covers/{str(today_str)}_covers.csv')
    
    
    
    # This will show the first few rows, use df to show the full DataFrame
    odds = pd.read_csv(f'../../data/odds/{str(today_str)}_odds.csv')
    merged_df = pd.merge(df, odds, on=['outcome_name'])
    merged_df = merged_df.drop(['Unnamed: 0'],axis=1)
    sorted_cols = ['id','outcome_name','commence_time','odds','Streak WL','Streak O/U','true_probability',
                   'ELO', 'Pitcher', 'Rest', 'Season WL', 'Season ERA','Season WHIP',
                   'Last 3 WL', 'Last 3 IP', 'Last 3 ERA', 'Last 3 WHIP','K', 'HR', 
                   'Team WLCS','home_team', 'away_team']
    merged_df = merged_df[sorted_cols]
    merged_df.to_csv(f'../../data/preview/{str(today_str)}_preview.csv', index=True)
    print('Merged DataFrame and saved to csv: Completed!')

def main():
    get_covers()
    print('Covers Data Retrieved and Saved!')

if __name__ == '__main__':
    main()
    
    
# %%
