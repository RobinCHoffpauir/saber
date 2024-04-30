import pandas as pd
import sqlite3
from collections import defaultdict

def load_data_from_db():
    """Load all game data from the SQLite database."""
    conn = sqlite3.connect('Final_Data.db')
    query = """
    SELECT Date, Tm, Opp, 'W/L' as win_loss
    FROM Final_Data
    ORDER BY Date ASC
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def initialize_elo():
    """Initialize ELO ratings for all teams as 1500."""
    return defaultdict(lambda: 1500)

def update_elo(winner_elo, loser_elo, k_factor=20):
    """Simple ELO rating adjustment."""
    expected_win = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
    change = k_factor * (1 - expected_win)
    return winner_elo + change, loser_elo - change

# Main processing script
elo_ratings = initialize_elo()
df = load_data_from_db()

# Iterate through games, updating ELOs
for index, row in df.iterrows():
    if row.win_loss.startswith('W'):
        winner, loser = row.Tm, row.Opp
    elif row.win_loss.startswith('L'):
        winner, loser = row.Opp, row.Tm
    else:
        continue  # Skip non-decisive games

    winner_elo, loser_elo = elo_ratings[winner], elo_ratings[loser]
    elo_ratings[winner], elo_ratings[loser] = update_elo(winner_elo, loser_elo)

# Optionally, save the updated ELO ratings back to the database or a CSV file
# Here, we'll demonstrate adding them to the DataFrame and then saving to a CSV
df['TmElo'] = df['Tm'].map(elo_ratings)
df['OppElo'] = df['Opp'].map(elo_ratings)

# Saving the DataFrame back to a new SQLite table or update the existing table
conn = sqlite3.connect('../../data/Final_Data.db')
df.to_sql('Final_Data_Updated', conn, if_exists='replace', index=False)
conn.close()
