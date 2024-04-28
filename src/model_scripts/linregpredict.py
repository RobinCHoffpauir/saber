import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Collect data
year = 2023
month = 4
pitching_data, outcomes = collect_data(year, month)

# Convert data to NumPy arrays
pitching_data = np.array(pitching_data)
outcomes = np.array(outcomes)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    pitching_data, outcomes, test_size=0.2, train_size=0.8, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean squared error:", mse)
print("R2 score:", r2)


#%%imports
import mlbgame
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#%% Function to collect pitching stats and game outcomes


def collect_data(year, month):
    days_in_month = mlbgame.calendar.monthrange(year, month)
    pitching_data = []
    outcomes = []

    for day in range(1, days_in_month[1] + 1):
        games = mlbgame.day(year, month, day)

        for game in games:
            game_id = game.game_id
            box_score = mlbgame.box_score(game_id)
            home_pitching = box_score.home_pitching
            away_pitching = box_score.away_pitching

            for pitcher in home_pitching + away_pitching:
                pitching_data.append([
                    pitcher.earned_runs,
                    pitcher.hits,
                    pitcher.home_runs,
                    pitcher.base_on_balls,
                    pitcher.strike_outs
                ])
                outcomes.append(pitcher.runs)

    return np.array(pitching_data), np.array(outcomes)


#%% Collect data
year = 2023
month = 4
pitching_data, outcomes = collect_data(year, month)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    pitching_data, outcomes, test_size=0.2, train_size=0.8, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean squared error:", mse)
print("R2 score:", r2)
