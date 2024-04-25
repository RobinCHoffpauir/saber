# Import required libraries
from bs4 import BeautifulSoup

import requests
import pandas as pd
import odds
from utils import *
from datetime import datetime



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

    # Create a DataFrame using table_data
    df = pd.DataFrame(table_data, columns=columns)
    df['outcome_name'] = df['Team'].map(team_names)
    df = df.drop(['Game Number','Team'],axis=1)
    # Show the resulting DataFrame
    
    # This will show the first few rows, use df to show the full DataFrame
    odds = pd.read_csv(f'./data/odds/{str(today_str)}_odds.csv')
    merged_df = pd.merge(df, odds, on=['outcome_name'])
    merged_df.index = merged_df['outcome_name']
    merged_df = merged_df.drop('outcome_name',axis=1)
    merged_df.to_csv(f'./data/preview/{str(today_str)}_preview.csv')
    print('Merged DataFrame and saved to csv: Completed!')

def main():
    get_covers()
    print('Covers Data Retrieved and Saved!')

if __name__ == '__main__':
    main()
    
    