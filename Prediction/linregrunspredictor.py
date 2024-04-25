# -*- coding: utf-8 -*-
"""
Created on Tue May  2 15:39:36 2023

@author: rhoffpauir
"""

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pybaseball as pyb
import pickle

df = pyb.fg_team_batting_data(2000, 2023, stat_columns=[
                              'OBP', 'SLG', 'ISO', 'OPS', 'WAR', 'wOBA', 'wRC', 'R'], split_seasons=True)

x = df.drop('R', axis=1)
x.index = x['Team']
x = x.drop(['teamIDfg', 'Team'], axis=1)
y = df['R']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

# Train the linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lr_model.predict(X_test)

# Calculate mean squared error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Calculate root mean squared error (RMSE)
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Calculate mean absolute error (MAE)
mae = mean_absolute_error(y_test, y_pred)

# Calculate R-squared (R2)
r2 = r2_score(y_test, y_pred)
# Print the metrics
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R-squared: {r2:.4f}")


# %%Feature Scaling and Cross-validation

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
LR_model = LinearRegression()
LR_model.fit(X_train_scaled, y_train)

# Perform cross-validation
cv_scores = cross_val_score(LR_model, X_train_scaled, y_train, cv=5)

# Print CV scores
print("CV Scores: ", cv_scores)

# Make predictions on the test set
y_pred = LR_model.predict(X_test_scaled)

# Calculate and print metrics as before...
# Print the metrics
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R-squared: {r2:.4f}")

# %% Decision Tree

dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train_scaled, y_train)

y_pred = dt_model.predict(X_test_scaled)

r2 = r2_score(y_test, y_pred)
# Print the metrics
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R-squared: {r2:.4f}")

# %%Random Forest

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

y_pred = rf_model.predict(X_test_scaled)
# Print the metrics
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R-squared: {r2:.4f}")

# %% Support Vector Machine

svr_model = SVR(kernel='linear')
svr_model.fit(X_train_scaled, y_train)

y_pred = svr_model.predict(X_test_scaled)
# Print the metrics
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R-squared: {r2:.4f}")
# %% Neural NEtwork

seq_model = Sequential()
seq_model.add(Dense(32, input_dim=X_train_scaled.shape[1], activation='relu'))
seq_model.add(Dense(16, activation='relu'))
seq_model.add(Dense(1))

seq_model.compile(loss='mean_squared_error', optimizer='adam')

seq_model.fit(X_train_scaled, y_train, epochs=50, batch_size=10)

y_pred = seq_model.predict(X_test_scaled)
# Print the metrics
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R-squared: {r2:.4f}")
