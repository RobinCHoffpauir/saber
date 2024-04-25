#%% imports and data loading
import pandas as pd, pybaseball as pyb, numpy as np, seaborn as sns, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from Betting.utils import *

# %%
pitchers = pyb.pitching_stats(2021, 2024)
# Assuming 'x' is your DataFrame
# Let's say you want to select all columns that start with a specific text and are in the list ['K%','K%+','SwStr%','Stuff+','Pitching+']
selected_columns = [col for col in pitchers.columns if 
                    (col.startswith('Stf+') or col.startswith('Pit+') or col.endswith('%')
                     or col.startswith('(sc)')) 
                    or col in ['Name','Team','K%','Stuff+','SwStr%','O-Swing%'
                               'Pitching+']]

# Now you can use the selected_columns list to select the desired columns
pitchers_selected = pitchers[selected_columns]
pitchers_selected = pitchers_selected.drop(['LOB%', 'LD%', 'GB%', 'FB%', 'IFFB%', 'IFH%', 
                                            'BUH%','Pull%', 'Cent%', 'Oppo%','Soft%', 
                                            'Med%', 'Hard%', 'TTO%', 'Barrel%', 'HardHit%',
                                            'PO%','KN%'],axis=1)
pitchers_selected = pitchers_selected.fillna('0')

# create function to pull team batting data from pybaseball and then parsing the columns to as closely
# match the Pitchers columns for analysis
batting_selected = pyb.fg_team_batting_data(2021,2024)
new_cols = [col for col in batting_selected.columns if col in selected_columns or col in ['Name','Team','K%','SwStr%','O-Swing%']]
batting_selected = batting_selected[new_cols]
batting_selected = batting_selected.fillna(0)

#%% Util functions to parse data for matchup predictions
def get_pitcher_data(name):
    output = pitchers_selected[pitchers_selected['Name'].str.contains(name, case=False)]
    return output

def get_opposing_team_data(team):
    output = batting_selected[batting_selected['Team'].str.contains(team, case=False)]
    return output
def get_features(pitcher_df, team_df):
    # This is a placeholder: adjust according to your specific feature selection logic
    combined_features = pd.concat([pitcher_df.iloc[:, :-1], team_df.iloc[:, :-1]], axis=1)
    return combined_features
def validate_data(pitcher_df, team_df):
    if pitcher_df.empty or team_df.empty:
        print("No data available for the given pitcher or team.")
        return False
    if len(pitcher_df) != len(team_df):
        print("Mismatch in the number of rows for pitcher and team data.")
        return False
    return True




def get_matchup_pred(pitcher_name, opp_team):
    pitcher_df = get_pitcher_data(pitcher_name)
    team_df = get_opposing_team_data(opp_team)

    if pitcher_df.empty or team_df.empty:
        print("No data available for the given pitcher or team.")
        return None

    # Ensure that the pitcher data and the team data have the same number of rows
    if len(pitcher_df) != len(team_df):
        print("Mismatch in the number of rows for pitcher and team data.")
        return None

    features = get_features(pitcher_df, team_df)
    target = pitcher_df['K%']  # Ensure target is correctly extracted

    if len(features) == 0:
        print("No features available for model training.")
        return None

    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    # Model training
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Evaluating model
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')

    return predictions# Example usage (This would be replaced by actual inputs in practice)
get_matchup_pred('Reid Detmers', "TBR")
    
#%%


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def get_pitcher_data(name):
    pitchers_selected = pitchers_selected[pitchers_selected['Name'] == name]
    return pitchers_selected

def get_opposing_team_data(team):
    batting_selected = batting_selected[batting_selected['Team'] == team]
    return batting_selected

def get_matchup_pred(pitcher_name, opp_team):
    pit = get_pitcher_data(pitcher_name)
    opp = get_opposing_team_data(opp_team)

    # Assuming 'x' is your DataFrame
    # Let's say you want to select all columns that start with a specific text and are in the list ['K%','K%+','SwStr%','Stuff+','Pitching+']
    selected_columns = [col for col in pitchers_selected.columns if
                            (col.startswith('Stf+') or col.startswith('Pit+') or col.endswith('%') or
                             col.endsswith('(sc)')) or col in ['Name','Team','K%','Stuff+','SwStr%','O-Swing%','Pitching+']]

    # Now you can use the selected_columns list to select the desired columns
    pitchers_selected = pitchers_selected[selected_columns]
    pitchers_selected = pitchers_selected.drop(['LOB%', 'LD%', 'GB%', 'FB%', 'IFFB%', 'IFH%',
                                                'BUH%','Pull%', 'Cent%', 'Oppo%','Soft%',
                                                'Med%', 'Hard%', 'TTO%', 'Barrel%', 'HardHit%',
                                                'PO%','KN%'],axis=1)
    pitchers_selected = pitchers_selected.fillna('0')

    # create function to pull team batting data from pybaseball and then parsing the columns to as closely
    # match the Pitchers columns for analysis
    batting_selected = pyb.fg_team_batting_data(2024)
    new_cols = [col for col in batting_selected.columns if col in selected_columns or col in ['Name','Team','K%','SwStr%','O-Swing%']]
    batting_selected = batting_selected[new_cols]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(pitchers_selected.drop('K', axis=1), pitchers_selected['K'], test_size=0.2, random_state=42)

    # Train the RandomForestRegressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions using the testing data
    y_pred = model.predict(X_test)

    # Calculate the mean squared error and R^2 score
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, r2
# %%
