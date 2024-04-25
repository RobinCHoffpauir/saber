import sqlite3
import pandas as pd, pybaseball as pyb, matplotlib.pyplot as plt, seaborn as sns, mlbgame as mlb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from datetime import datetime, timedelta
import logging
import os
import pickle
import numpy as np
from collections import defaultdict
import covers, scores, odds
today = datetime.today()
yesterday = (today - timedelta(days=1))
DATE = today.strftime('%Y-%m-%d')

pyb.cache.enable()
pd.options.display.max_rows = None
pd.options.display.max_columns = None

def correlation_heatmap(df, method='pearson'):
    """
    Generate a heatmap for the correlation matrix of the dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe for which the correlation matrix is to be generated.
    method : str, optional
        The method to be used for calculating the correlation coefficients. The default is 'pearson'. Other valid methods include 'kendall', 'spearman', and 'mutual_info_classif'.

    Returns
    -------
    None
        This function does not return any value. It generates and displays a heatmap using matplotlib and seaborn.

    Raises
    ------
    ValueError
        If the input dataframe does not contain at least two numerical columns, a ValueError will be raised.

    Example
    -------
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Assuming 'df' is a DataFrame containing numerical data
    correlation_heatmap(df, method='pearson')
    """
    corr_matrix = df.corr(method=method)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.show()    
    
def odds_to_probability(odds):
    """
    Converts odds to probability.

    Parameters:
    - odds (float): The odds value to be converted.

    Returns:
    - float: The probability value corresponding to the given odds.

    The function calculates the probability based on the given odds. If the odds are negative, it calculates the probability using the formula:
    `prob = (-odds) / ((-odds) + 100)`. If the odds are positive, it calculates the probability using the formula:
    `prob = 100 / (odds + 100)`.

    Example:
    ```
    >>> odds_to_probability(-150)
    0.5555555555555555
    >>> odds_to_probability(150)
    0.4
    ```
    """
    if odds < 0:
        prob = (-odds) / ((-odds) + 100)
    else:
        prob = 100 / (odds + 100)
    return prob
def expected_outcome_elo(rating1, rating2):
    """
    Calculates the expected outcome of a matchup based on the ratings of the two teams.

    Parameters:
    - rating1 (float): The rating of the first team.
    - rating2 (float): The rating of the second team.

    Returns:
    - float: The expected outcome of the matchup, where 1 represents a win for the first team, 0 represents a draw, and 0 < x < 1 represents a win for the second team.

    The function uses the Elo rating system to calculate the expected outcome of a matchup between two teams. The Elo rating system is a method for calculating the relative skill levels of players in two-player games such as chess. In this implementation, the expected outcome is calculated as the probability that the first team will win the matchup.

    Example:
    ```
    >>> expected_outcome(1600, 1500)
    0.5555555555555555
    ```
    """
    return 1 / (1 + 10 ** ((rating2 - rating1) / 400))

# Determine the winner of each matchup based on the expected outcome
def determine_winner(home_prob, away_prob):
    """
    Determines the winner of a matchup based on the expected outcome.

    Parameters:
    - home_prob (float): The probability of the home team winning.
    - away_prob (float): The probability of the away team winning.

    Returns:
    - str: The winner of the matchup, either 'Home', 'Away', or 'Tie'.

    The function compares the probabilities of the home and away teams winning and returns the team with the higher probability as the winner. If both probabilities are equal, the function returns 'Tie'.

    Example:
    ```
    >>> determine_winner(0.6, 0.4)
    'Home'
    >>> determine_winner(0.4, 0.6)
    'Away'
    >>> determine_winner(0.5, 0.5)
    'Tie'
    ```
    """
    if home_prob > away_prob:
        return 'Home'
    elif away_prob > home_prob:
        return 'Away'
    else:
        return 'Tie'

def generate_predictions(odds_data):
    """
    Generates predictions for each matchup based on the expected outcome.

    Parameters:
    - odds_data (DataFrame): A DataFrame containing odds data for each matchup.

    Returns:
    - DataFrame: A DataFrame containing the predicted outcomes for each matchup.

    The function iterates through each row of the input DataFrame, calculates the expected outcome for each matchup, and determines the winner based on the expected outcome. The predicted outcomes are then combined using the `pd.concat` method, which concatenates the input DataFrames along the specified axis. The resulting DataFrame is returned as the result.

    Example:
    ```
    import pandas as pd
    # Load the odds data generated by "odds.py" into a Pandas DataFrame
    odds_file = 'data/odds/2024-01-01_odds.csv'
    odds_data = pd.read_csv(odds_file)
    # Generate predictions for each matchup
    predictions_data = generate_predictions(odds_data)
    # Save the predicted outcomes to a CSV file
    predictions_data.to_csv('data/predictions/2024-01-01_predictions.csv', index=False)
    ```
    """
    predictions_data = pd.DataFrame(columns=['Home', 'Away', 'Winner'])

    for index, row in odds_data.iterrows():
        home_prob = row['probability']
        away_prob = 1 - home_prob
        winner =generate_predictions(odds_data)

def update_elo(winner_elo: float, loser_elo: float, k_factor: float) -> tuple[float, float]:
    """
    Update the Elo ratings of two players based on the outcome of a matchup.

    Args:
        winner_elo (float): The Elo rating of the winner of the matchup.
        loser_elo (float): The Elo rating of the loser of the matchup.
        k_factor (float): The factor used to calculate the change in Elo rating.

    Returns:
        tuple[float, float]: A tuple containing the updated Elo ratings of the two players.

    The Elo rating system is a method for calculating the relative skill levels of players in two-player games. The update_elo function calculates the expected win probability of the winner based on the Elo ratings of the two players, and then updates the Elo ratings of the two players based on the change in rating. The k_factor parameter controls the strength of the influence of the expected win probability on the Elo rating. A higher k_factor means that the expected win probability has a greater influence on the Elo rating.

    Example:
    >>> update_elo(1600, 1500, 32)
    (1632.0, 1468.0)
    """
    expected_win = expected_outcome_elo(winner_elo, loser_elo)
    change_in_elo = k_factor * (1 - expected_win)
    winner_elo += change_in_elo
    loser_elo -= change_in_elo
    return winner_elo, loser_elo

def load_elo_ratings(year: int) -> dict:
    """
    Loads Elo ratings for each team in the specified year.

    Parameters:
    year (int): The year for which the Elo ratings are to be loaded.

    Returns:
    dict: A dictionary containing the Elo ratings for each team in the specified year. If no Elo ratings file exists for the specified year, a default dictionary with all ratings set to 1500 is returned.

    Raises:
    FileNotFoundError: If the Elo ratings file for the specified year does not exist.

    Example:
    >>> elo_dict = load_elo_ratings(2021)
    >>> print(elo_dict)
    {'BOS': 1500, 'NYY': 1500, 'TBR': 1500, 'KCR': 1500, 'CHW': 1500, 'BAL': 1500, 'CLE': 1500, 'MIN': 1500, 'DET': 1500, 'HOU': 1500, 'LAA': 1500, 'SEA': 1500, 'TEX': 1500, 'OAK': 1500, 'WSN': 1500, 'MIA': 1500, 'ATL': 1500, 'NYM': 1500, 'PHI': 1500, 'CHC': 1500, 'MIL': 1500, 'STL': 1500, 'PIT': 1500, 'CIN': 1500, 'LAD': 1500, 'ARI': 1500, 'COL': 1500, 'SDP': 1500, 'SFG': 1500, 'TOR': 1500}
    """
    filename = f"{year}_elo.csv"
    if os.path.exists(filename):
        return pd.read_csv(filename).set_index('Team')['ELO'].to_dict()
    else:
        return defaultdict(lambda: 1500)  # Default ELO rating

def save_elo_ratings(elo_dict, year):
    """
    Save the Elo ratings for each team in the specified year to a CSV file.

    Parameters:
    elo_dict (dict): A dictionary containing the Elo ratings for each team in the specified year.
    year (int): The year for which the Elo ratings are to be saved.

    Returns:
    None

    Raises:
    FileNotFoundError: If the Elo ratings file for the specified year does not exist.

    Example:
    >>> elo_dict = {'BOS': 1500, 'NYY': 1500, 'TBR': 1500, 'KCR': 1500, 'CHW': 1500, 'BAL': 1500, 'CLE': 1500, 'MIN': 1500, 'DET': 1500, 'HOU': 1500, 'LAA': 1500, 'SEA': 1500, 'TEX': 1500, 'OAK': 1500, 'WSN': 1500, 'MIA': 1500, 'ATL': 1500, 'NYM': 1500, 'PHI': 1500, 'CHC': 1500, 'MIL': 1500, 'STL': 1500, 'PIT': 1500, 'CIN': 1500, 'LAD': 1500, 'ARI': 1500, 'COL': 1500, 'SDP': 1500, 'SFG': 1500, 'TOR': 1500}
    >>> save_elo_ratings(elo_dict, 2021)
    """
    filename = f"Betting/data/elo/{year}_elo.csv"
    pd.DataFrame(list(elo_dict.items()), columns=['Team', 'ELO']).to_csv(filename, index=False)

def get_elo(team, elo_dict, default_elo=1500):
    """
    Get the Elo rating for a given team from the provided dictionary. If the team is not found in the dictionary, return the default Elo rating.

    Parameters:
    - team (str): The name of the team for which the Elo rating is requested.
    - elo_dict (dict): A dictionary containing the Elo ratings for each team.
    - default_elo (int, optional): The default Elo rating to return if the requested team is not found in the dictionary. Defaults to 1500.

    Returns:
    - int: The Elo rating for the requested team, or the default Elo rating if the team is not found in the dictionary.

    Example:
    ```
    # Assuming 'elo_dict' is a dictionary containing Elo ratings for teams
    print(get_elo('BOS', elo_dict))  # Output: Elo rating for Boston Red Sox
    print(get_elo('NYM', elo_dict))  # Output: Elo rating for New York Mets
    print(get_elo('NYJ', elo_dict))  # Output: Default Elo rating (1500) as 'NYJ' is not in the dictionary
    ```
    """
    return elo_dict.get(team, default_elo)

def setup_logging():
    """Setup basic logging configuration."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_model(model, filename):
    """Save the trained model to a file."""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    """Load a trained model from a file."""
    with open(filename, 'rb') as f:
        model = pickle.load(f)
        

def load_data(type='batting', year=today.year):
    """
    Loads baseball stats data from the pybaseball library.

    Parameters:
    type (str): The type of stats to load, either 'pitching' or 'batting'.
    year (int): The year of the stats to load. Defaults to the current year.

    Returns:
    pandas.DataFrame: A DataFrame containing the requested stats data.

    Raises:
    ValueError: If an invalid type is provided.

    Example:
    >>> pitching_data = load_data(type='pitching', year=2021)
    >>> batting_data = load_data(type='batting', year=2020)
    """
    if type == 'pitching':
        return pyb.team_pitching(year)
    elif type == 'batting':
        return pyb.team_batting(year)
    else:
        raise ValueError("Invalid type provided. Please choose 'pitching' or 'batting'.")
    
def split_data(df, target_variable):
    """
    Split data into features and target, then into training and testing sets.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe containing all the features and target variable.
    target_variable : str
        The name of the target variable in the dataframe.

    Returns
    -------
    str
        A message indicating that the data has been split into training and testing sets.

    Raises
    ------
    ValueError
        If the target_variable is not found in the dataframe.

    Example
    -------
    >>> df = load_data(type='batting', year=2020)
    >>> split_data(df, 'Runs')
    'train/test/split complete'
    """
    X = df.drop(target_variable, axis=1)
    y = df[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return "train/test/split complete"

def fit_lr_model(X_train, y_train):
    """
    Fit a linear regression model to the training data.

    Parameters
    ----------
    X_train : pandas.DataFrame
        The input dataframe containing the features for the training set.
    y_train : pandas.Series
        The input dataframe containing the target variable for the training set.

    Returns
    -------
    sklearn.linear_model.LinearRegression
        A trained linear regression model.

    Raises
    ------
    ValueError
        If the input dataframes are not of the correct types or shapes.

    Example
    -------
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.model_selection import train_test_split
    >>> import pandas as pd
    >>> # Assuming 'df' is a DataFrame containing the features and target variable
    >>> X_train, X_test, y_train, y_test = train_test_split(df.drop('target_variable', axis=1), df['target_variable'], test_size=0.2, random_state=42)
    >>> model = fit_lr_model(X_train, y_train)
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using Mean Squared Error and R2score.

    Parameters
    ----------
    model : sklearn.linear_model.LinearRegression
        A trained linear regression model.
    X_test : pandas.DataFrame
        The input dataframe containing the features for the testing set.
    y_test : pandas.Series
        The input dataframe containing the target variable for the testing set.

    Returns
    -------
    tuple
        A tuple containing the Mean Squared Error (mse) and R2score (r2) of the model.

    Raises
    ------
    ValueError
        If the input dataframes are not of the correct types or shapes.

    Example
    -------
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.model_selection import train_test_split
    >>> import pandas as pd
    >>> # Assuming 'df' is a DataFrame containing the features and target variable
    >>> X_train, X_test, y_train, y_test = train_test_split(df.drop('target_variable', axis=1), df['target_variable'], test_size=0.2, random_state=42)
    >>> model = fit_lr_model(X_train, y_train)
    >>> mse, r2 = evaluate_model(model, X_test, y_test)
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mse, r2

# Function to calculate winning percentage
def calculate_winning_percentage(record):
    """
    Calculate the winning percentage of a team based on their wins and losses.

    Parameters
    ----------
    record : str
        A string representing the team's record in the format "wins-losses".

    Returns
    -------
    float
        The winning percentage of the team, as a float between 0 and 1.

    Raises
    ------
    ValueError
        If the input record is not in the format "wins-losses".

    Example
    -------
    >>> calculate_winning_percentage("100-50")
    0.6666666666666667
    >>> calculate_winning_percentage("50-100")
    0.3333333333333335
    """
    wins, losses = map(int, record.split('-'))
    total_games = wins + losses
    if total_games == 0:  # Avoid division by zero
        return 0
    return wins / total_games

def remove_vig(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function removes the vig (vigorish) from the probabilities in a DataFrame.

    Parameters:
    - df (pd.DataFrame): A DataFrame containing the odds data for each matchup.

    Returns:
    - pd.DataFrame: A DataFrame with the probabilities normalized to remove the vig.

    The function calculates the sum of the probabilities including vig, and then normalizes the probabilities by dividing each probability by the total sum. This effectively removes the vig from the probabilities.

    Example:
    ```
    import pandas as pd
    # Load the odds data generated by "odds.py" into a Pandas DataFrame
    odds_file = 'data/odds/2024-01-01_odds.csv'
    odds_data = pd.read_csv(odds_file)
    # Remove the vig from the probabilities
    true_probabilities = remove_vig(odds_data)
    ```
    """
    # Calculate the sum of the probabilities including vig
    total_probability = df['imp_probability'].sum()

    # Normalize the probabilities to remove the vig
    df['true_probability'] = df['imp_probability'] / total_probability
    
    return df

def get_roster_dfs():
    """
    This function retrieves the roster data for each team from the provided SQLite database.

    Parameters:
    - None: This function does not require any parameters.

    Returns:
    - pd.DataFrame: A DataFrame containing the roster data for each team.

    The function iterates through each team in the 'teams' list, connects to the SQLite database, executes a SQL query to retrieve the roster data for the current team, and then creates a Pandas DataFrame from the retrieved data. The resulting DataFrame is then returned.

    Example:
    ```
    import pandas as pd

    # Assuming 'get_roster_dfs' function is defined and working correctly

    # Get roster data for all teams
    roster_data = get_roster_dfs()

    # Print the first 5 rows of the roster data for the first team
    print(roster_data.head(5))
    ```
    """
    for team in teams:
        con = sqlite3.connect("Betting/data/rosters/mlb_rosters.db")
        cursor = con.cursor()
        query = f"SELECT * FROM {team}"
        cursor.execute(query)
        cols = [description[0] for description in cursor.description]
        data = cursor.fetchall()
        team = pd.DataFrame(data=data, columns= cols) 


################################################################################
#Variables
global divisions, teams, team_colors

divisions = {
    "AL East": ["BOS", "NYY", "TBR", "TOR", "BAL"],
    "AL Central": ["CLE", "MIN", "KCR", "CHW", "DET"],
    "AL West": ["HOU", "LAA", "SEA", "TEX", "OAK"],
    "NL East": ["WSN", "MIA", "ATL", "NYM", "PHI"],
    "NL Central": ["CHC", "MIL", "STL", "PIT", "CIN"],
    "NL West": ["LAD", "ARI", "COL", "SDP", "SFG"],
}

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

teams = ['BOS',
         'NYY',
         'TBR',
         'KCR',
         'CHW',
         'BAL',
         'CLE',
         'MIN',
         'DET',
         'HOU',
         'LAA',
         'SEA',
         'TEX',
         'OAK',
         'WSN',
         'MIA',
         'ATL',
         'NYM',
         'PHI',
         'CHC',
         'MIL',
         'STL',
         'PIT',
         'CIN',
         'LAD',
         'ARI',
         'COL',
         'SDP',
         'SFG',
         'TOR']
team_colors = {
    "BOS": "#BD3039",
    "NYY": "#003087",
    "TBR": "#8FBCE6",
    "KCR": "#BD9B60",
    "CHW": "#27251F",
    "BAL": "#DF4601",
    "CLE": "#E31937",
    "MIN": "#002B5C",
    "DET": "#FA4616",
    "HOU": "#EB6E1F",
    "LAA": "#BA0021",
    "SEA": "#005C5C",
    "TEX": "#003278",
    "OAK": "#003831",
    "WSN": "#14225A",
    "MIA": "#FF6600",
    "ATL": "#13274F",
    "NYM": "#002D72",
    "PHI": "#E81828",
    "CHC": "#0E3386",
    "MIL": "#B6922E",
    "STL": "#C41E3A",
    "PIT": "#FDB827",
    "CIN": "#C6011F",
    "LAD": "#005A9C",
    "ARI": "#A71930",
    "COL": "#33006F",
    "SDP": "#002D62",
    "SFG": "#FD5A1E",
    "TOR": "#134A8E",
}

