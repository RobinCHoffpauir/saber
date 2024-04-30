
#%%
import requests
from datetime import datetime, timedelta
import pandas as pd
from csv import DictReader
from helpers import *
today = datetime.today()



# An api key is emailed to you when you sign up to a plan
# Get a free API key at https://api.the-odds-api.com/
API_KEY = '1628e7b26fd53acf9eef611ac3466368'


# Sport key
# Alternatively use 'upcoming' to see the next 8 games across all sports
SPORT = 'baseball_mlb'

# Bookmaker regions
# uk | us | us2 | eu | au. Multiple can be specified if comma delimited.
REGIONS = 'us2'

# h2h | spreads | totals. Multiple can be specified if comma delimited
# Note only featured markets (h2h, spreads, totals) are available with the odds endpoint.
MARKETS = 'h2h'

# Odds format
# decimal | american
ODDS_FORMAT = 'american'

# Date format
# iso | unix
DATE_FORMAT = 'iso'

# Bookmaker
# hardrockbet
BOOKMAKER = 'hardrockbet'

DATE = today.strftime('%Y-%m-%d')

def get_data_odds():
    odds_response = requests.get(f'https://api.the-odds-api.com/v4/sports/{SPORT}/odds', params={
        'api_key': API_KEY,
        'regions': REGIONS,
        'markets': MARKETS,
        'oddsFormat': ODDS_FORMAT,
        'dateFormat': DATE_FORMAT,
        'bookmakers' : BOOKMAKER,
        'date': DATE
    })
    global api_data
    api_data = []

    # Check if the request was successful
    if odds_response.status_code != 200:
        print(f'Failed to get odds: status_code {odds_response.status_code}, response body {odds_response.text}')

    else:
        odds_json = odds_response.json()
        api_data = odds_response.json()
      # Check the usage quota
        print('Remaining requests', odds_response.headers['x-requests-remaining'])
        print('Used requests', odds_response.headers['x-requests-used'])

    # Initialize a list to store your processed data
    processed_data = []

    # Loop through each event in the API data
    for event in api_data:
        # Extract event details
        event_details = {
            'id': event['id'],
            'sport_title': event['sport_title'],
            'commence_time': event['commence_time'],
            'home_team': event['home_team'],
            'away_team': event['away_team'],
        }

        # Check if there are bookmakers for this event
        if event['bookmakers']:
            # Loop through each bookmaker in the event
            for bookmaker in event['bookmakers']:
                # Check if there are markets for this bookmaker
                if bookmaker['markets']:
                    # Loop through each market for the bookmaker
                    for market in bookmaker['markets']:
                        # Check if the market key is 'h2h' for head-to-head odds
                        if market['key'] == 'h2h':
                            # Loop through each outcome in the market to extract odds
                            for outcome in market['outcomes']:
                                # Prepare a dictionary with event, bookmaker, and odds details
                                data_row = {**event_details,
                                            'bookmaker': bookmaker['title'],
                                            'outcome_name': outcome['name'],
                                            'odds': outcome['price']}
                                # Append the dictionary to the processed data list
                                processed_data.append(data_row)
        else:
            # If no bookmakers, still append the event details with empty values for the rest
            processed_data.append({**event_details, 'bookmaker': None, 'outcome_name': None, 'odds': None})

    # Create a DataFrame from the processed data
    df = pd.DataFrame(processed_data)

    # First, let's drop rows where odds are NaN as these do not contain valid bookmaker odds
    df = df.dropna(subset=['odds'])

    # Apply the conversion function to each odds value
    df['imp_probability'] = df['odds'].apply(odds_to_probability) #calculate using function by .apply and save odds_data to csv
 
    # Apply the function to each game group
    df = df.groupby('id').apply(remove_vig)
    df['imp_probability']=df['imp_probability'].apply(lambda x: f"{x: .2f}%")
    df['true_probability']=df['true_probability'].apply(lambda x: f"{x: .2f}%")
   
    #drop unwanted columns
    df = df.drop(['sport_title','bookmaker'],axis=1) 
   
    #transform to datetime
    df['commence_time'] = pd.to_datetime(df['commence_time'])
   
    #save data to csv
    df.to_csv(f'../../data/odds/{str(DATE)}_odds.csv', index=False)
       
       
def main():
    get_data_odds()
    
    
if __name__ == '__main__':
    main()

# %%
