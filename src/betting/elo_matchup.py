import pandas as pd
from helpers import *
# Sample DataFrame setup

data = pd.read_csv('../../data/elo/2024_elo.csv')
df = pd.DataFrame(data)

# Function to calculate win probability
def win_probability(R_A, R_B):
    return 1 / (1 + 10 ** ((R_B - R_A) / 400))

# Function to convert probability to American odds
def to_american_odds(probability):
    if probability >= 0.5:
        return -100 / probability
    else:
        return 100 / (1 - probability)

teamA = input('Enter Team A: ')
teamB = input('Enter Team B: ')
# Example: Calculate odds for TeamA vs TeamB
R_A = df[df['Team'] == teamA]
R_B = df[df['Team'] == teamB]

prob_A = win_probability(R_A['ELO'].values, R_B['ELO'].values)
american_odds_A = to_american_odds(float(prob_A))

prob_B = win_probability(R_B['ELO'].values, R_A['ELO'].values)
american_odds_B = to_american_odds(float(prob_B))

print(f"American Odds for Team A: {american_odds_A:.2f}")
print(f"American Odds for Team B: {american_odds_B:.2f}")
