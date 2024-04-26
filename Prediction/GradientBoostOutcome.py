
#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
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

# merging as previously shown
df_covers = pd.read_csv('./Betting/data/preview/2024-04-16_covers.csv')
# reading the probabilities from CSV
df_probs = pd.read_csv('./Betting/data/probabilities/2024-04-16_probabilities.csv')
# map the names of the outcomes to their abbreviations
abbrev_dict = {value: key for key, value in team_names.items()}
# Function to map full names to abbreviations
def map_to_abbrev(name):
    """
    This function takes a full team name as input and returns the corresponding abbreviation.
    If the full team name is not found in the dictionary, it defaults to 'Unknown'.

    Args:
    name (str): The full team name to be mapped to its abbreviation.

    Returns:
    str: The corresponding abbreviation of the input team name.

    Raises:
    KeyError: If the full team name is not found in the dictionary.
    """
    return abbrev_dict.get(name, 'Unknown')  # Default to 'Unknown' if not found

# Applying the function to create a new column with abbreviations
df_probs['Team'] = df_probs['outcome_name'].apply(map_to_abbrev)

merged_df = pd.merge(df_covers, df_probs, on=['Team'])

#%%
# Convert categorical data to numerical data
le = LabelEncoder()
categories = merged_df[['Pitcher','Last 3 WL','Streak WL','Streak O/U','Season WL','Team WLCS']]
for cat in categories.columns:
    merged_df[cat] = le.fit_transform(merged_df[cat])
merged_df.index = merged_df['Team']
# Select features and targets
X = merged_df.drop(['true_probability','Team','id','outcome_name'], axis=1)
y = ((merged_df['true_probability'])) # Assuming true_probability > 0.5 means Team A wins

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the logistic regression model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)
model.fit(X_train, y_train)
# Predicting the test set results
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = (y_test - y_pred)
#conf_matrix = confusion_matrix((y_test*100), (y_pred*100))

print("Accuracy:", accuracy)
#print("Confusion Matrix:\n", conf_matrix)
#%%dump model
# joblib.dump(model, 'data/models/GradientBoostModel.pkl')
# %%
