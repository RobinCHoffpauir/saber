from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from pybaseball import fg_batting_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import joblib
scaler = MinMaxScaler()
# Fetch hitter data from 2010 to 2021
start_year = 2021
end_year = 2024
hitter_data = pd.concat([fg_batting_data(year, ind=1) for year in range(start_year, end_year + 1)])

# Calculate team-level statistics by summing the individual player stats
team_stats = hitter_data.groupby(['Team']).agg({
    'xwOBA': 'mean',
    'Off': 'mean',
    'L-WAR': 'mean',
    'R': 'sum',
    'wRC+':'sum',
    'ISO+':'mean',
    'Barrels': 'mean',
    '+WPA': 'sum',
}).reset_index()


# Extract the features (wOBA, wRC, OPS, WAR, ISO) and targe
X = team_stats[['xwOBA','Off','L-WAR','wRC+','ISO+','Barrels','+WPA']]
y = team_stats['R']
X_scaled = scaler.fit_transform(X)

def calculate_composite_metric(row, weights):
    return sum(row[stat] * weight for stat, weight in weights.items())


# %%
# Split the data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# Perform multiple linear regression
reg = LinearRegression()


# Train the model
reg.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = reg.predict(X_test)

# The coefficients
print('Coefficients: \n', reg.coef_)

# The mean squared error
print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))

# The root mean squared error
print('Root mean squared error: %.2f' %
      mean_squared_error(y_test, y_pred, squared=False))

# The mean absolute error
print('Mean absolute error: %.2f' % mean_absolute_error(y_test, y_pred))

# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination (R^2): %.2f' % r2_score(y_test, y_pred))

# %%create a linear regression object
lm = LinearRegression()

# Cross-validate
scores = cross_val_score(lm, X, y, cv=5, scoring='neg_mean_squared_error')

# The scores returned are negative because cross_val_score assumes a utility function (higher is better).
# Therefore, it multiplies the outputs by -1.
# For our purposes, we want to convert these back into positive numbers, which we can do by simply flipping the sign.
mse_scores = -scores

# calculate the mean of the MSE scores
mean_mse = np.mean(mse_scores)

print(f'5-fold cross-validated Mean Squared Error: {mean_mse:.2f}')
# %%
# Display the weights (coefficients) for each stat
weights = dict(zip(X.columns, reg.coef_))
print("Weights for each stat:")
for stat, weight in weights.items():
    print(f"{stat}: {weight:.4f}")

team_stats['composite_metric'] = team_stats.apply(
    lambda row: calculate_composite_metric(row, weights), axis=1)
# %%

# assuming you have your actual test values 'y_test' and predicted values 'y_pred'
y_pred = reg.predict(X_test)

print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: ", mean_squared_error(
    y_test, y_pred, squared=False))
print("Mean Absolute Error: ", mean_absolute_error(y_test, y_pred))
print("R-squared: ", r2_score(y_test, y_pred))
# %% pickle the model for later use

#joblib.dump(lm, 'Betting/data/models/linreg_model.pkl')

# %%
team_stats_sorted = team_stats.sort_values(
    ['composite_metric'], ascending=[False])
print(team_stats_sorted[['Team', 'composite_metric', 'R']])
# %%

sns.scatterplot(data=team_stats, x='composite_metric', y='R')
plt.xlabel('Composite Metric')
plt.ylabel('Runs Scored')
plt.show()
