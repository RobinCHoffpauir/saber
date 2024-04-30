#%%
from pybaseball import fg_team_fielding_data, fg_team_pitching_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

start_year = 2021
end_year = 2024
# Fetch team pitching data from 2010 to 2021
team_pitching_data = pd.concat([fg_team_pitching_data(season) for season in range(start_year, end_year + 1)])

# Fetch team fielding data from 2010 to 2021
team_fielding_data = pd.concat([fg_team_fielding_data(season) for season in range(start_year, end_year + 1)])

# Merge team stats (hitting, pitching, and fielding)
team_stats = pd.DataFrame()
team_stats = pd.merge(team_pitching_data,team_fielding_data, left_on=['Season', 'Team'], right_on=['Season','Team'], suffixes=('', '_pitching'))
#team_stats = team_stats.merge(team_fielding_data, left_on=['Season', 'Team'], right_on=['Season', 'Team'])

# Rename columns for clarity
team_stats.rename(columns={'WAR': 'WAR_hitting', 'WAR_pitching': 'WAR_pitching', 'DRS': 'DRS_fielding'}, inplace=True)
#%%
team_stats['RDiff'] = team_stats['R'] - team_stats['RA']
X = team_stats.drop(columns=['RDiff','R', 'RA'], axis=1)
y = team_stats['RDiff']


#%%


# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
reg = LinearRegression().fit(X_train, y_train)

# Make predictions using the testing data
y_pred = reg.predict(X_test)

# Calculate the mean squared error and R^2 score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")
