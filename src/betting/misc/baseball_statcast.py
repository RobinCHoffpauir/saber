#!/usr/bin/env python
# coding: utf-8

# %%imports

import plotly.express as px
import numpy as np
import pybaseball as pyb
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import sqlite3

pyb.cache.enable()
today = dt.datetime.today().isoformat()[:10]


# %% data collection and parsing
teams = ['BOS', 'NYY', 'TBR', 'TOR', 'BAL', 'CLE', 'MIN', 'KCR', 'CHW',
         'DET', 'HOU', 'LAA', 'SEA', 'TEX', 'OAK', 'WSN', 'MIA', 'ATL',
         'NYM', 'PHI', 'CHC', 'MIL', 'STL', 'PIT', 'CIN', 'LAD', 'ARI',
         'COL', 'SDP', 'SFG']


# df = pd.read_csv('data/statcast.csv').drop('Unnamed: 0', axis=1)
# # Filter in_play and not_in_play DataFrames
# in_play = df.loc[df['type'] == 'X']
# not_in_play = df.loc[df['type'] != 'X']

# # Filter hits, strikes, and balls DataFrames
# hits = in_play.loc[in_play['events'].isin(
#     ['single', 'double', 'triple', 'home_run'])]
# strikes = not_in_play.loc[not_in_play['type'] == 'S']
# balls = not_in_play.loc[not_in_play['type'] == 'B']

# # Filter lefty_st, righty_st, st_fastballs, and st_offspeed DataFrames
# lefty_st = strikes.loc[strikes['p_throws'] == 'L']
# righty_st = strikes.loc[strikes['p_throws'] == 'R']
# st_fastballs = strikes.loc[strikes['pitch_name'].isin(
#     ['4-Seam Fastball', 'Cutter', 'Sinker', 'Fastball', 'Split-Finger'])]
# st_offspeed = strikes.loc[strikes['pitch_name'].isin(
#     ['Curveball', 'Changeup', 'Slider', 'Knuckle Curve', 'Eephus'])]

# # Filter strikeouts DataFrame
# strikeouts = not_in_play.loc[not_in_play['events'].isin(
#     ['strikeout', 'strikeout_double_play'])]
# %%import data from sqlite db
# Create a SQLite3 database connection
conn = sqlite3.connect('.\data\statcast_events.db')

# Read all the tables into DataFrames
hits_df = pd.read_sql('SELECT * FROM hits', conn)
strikes_df = pd.read_sql('SELECT * FROM strikes', conn)
balls_df = pd.read_sql('SELECT * FROM balls', conn)
lefty_st_df = pd.read_sql('SELECT * FROM lefty_st', conn)
righty_st_df = pd.read_sql('SELECT * FROM righty_st', conn)
st_fastballs_df = pd.read_sql('SELECT * FROM st_fastballs', conn)
st_offspeed_df = pd.read_sql('SELECT * FROM st_offspeed', conn)
strikeouts_df = pd.read_sql('SELECT * FROM strikeouts', conn)

# %%
con = sqlite3.connect('.\data\2022_gamelogs.db')
for team in teams:
    team = pd.read_sql(f'SELECT * FROM {team}', con)

# %% build dictionary to hold each team's schedule and results for chosen season
team_dfs = {}  # Create an empty dictionary to store DataFrames for each team

for x in teams:  # using for loop to iterate through teams list
    team_dfs[x] = pyb.team_game_logs(2022, x)

for team, df in team_dfs.items():
    # Split the 'Rslt' column by ',' and create new columns for "Rslt" and "Score"
    split_columns = df['Rslt'].str.split(',', expand=True)

    # Rename the new columns
    split_columns.columns = ['Rslt', 'Score']

    # Concatenate the new columns with the original DataFrame
    team_dfs[team] = pd.concat(
        [df.drop('Rslt', axis=1), split_columns], axis=1)

# %% Create a DataFrame for runs scored and runs allowed per game for all teams
runs_scored_vs_allowed = pd.concat([df.assign(Team=team,
                                              Runs_Scored=df['Score'].apply(
                                                  lambda x: int(x.split('-')[0])),
                                              Runs_Allowed=df['Score'].apply(lambda x: int(x.split('-')[1])))
                                    for team, df in team_dfs.items()])

# Create a pivot table for heatmap
pivot_table = runs_scored_vs_allowed.groupby(
    ['Runs_Scored', 'Runs_Allowed']).size().reset_index(name='Count')
pivot_table = pivot_table.pivot(
    'Runs_Scored', 'Runs_Allowed', 'Count').fillna(0)

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(pivot_table, cmap='coolwarm', annot=True, fmt='.0f')
plt.xlabel('Runs Allowed')
plt.ylabel('Runs Scored')
plt.title('Runs Scored vs. Runs Allowed per Game for All Teams in 2022')
plt.show()


# %%

# Calculate average runs scored, runs allowed, and run differential for each team
team_summary = runs_scored_vs_allowed.groupby('Team').agg(
    {'Runs_Scored': 'mean', 'Runs_Allowed': 'mean'}).reset_index()
team_summary['Run_Differential'] = team_summary['Runs_Scored'] - \
    team_summary['Runs_Allowed']

# Create a color map for the teams
team_color_map = {team: i for i, team in enumerate(teams)}
team_summary['color'] = team_summary['Team'].map(team_color_map)

# Plot parallel coordinates plot
fig = px.parallel_coordinates(team_summary, dimensions=['Runs_Scored', 'Runs_Allowed', 'Run_Differential'], color='color', color_continuous_scale=px.colors.qualitative.Plotly, labels={
                              'Runs_Scored': 'Avg. Runs Scored', 'Runs_Allowed': 'Avg. Runs Allowed', 'Run_Differential': 'Avg. Run Differential'})
fig.update_layout(
    title='Average Runs Scored, Runs Allowed, and Run Differential for Each Team in 2022')
plt.show()


# %% In[38]
# Plot the count of balls and strikes for left-handed and right-handed pitchers
plt.figure(figsize=(10, 6))
sns.countplot(x="p_throws", hue="type", data=strikes)
plt.title("Count of Balls and Strikes by Pitcher Handedness")
plt.xlabel("Pitcher Handedness")
plt.ylabel("Count")
plt.show()

# Plot the distribution of pitch speeds for fastballs and offspeed pitches
plt.figure(figsize=(10, 6))
sns.kdeplot(x="release_speed", hue="pitch_name", data=st_fastballs)
sns.kdeplot(x="release_speed", hue="pitch_name", data=st_offspeed)
plt.title("Distribution of Pitch Speeds by Pitch Type")
plt.xlabel("Pitch Speed (mph)")
plt.ylabel("Density")
plt.show()

# Plot the count of strikeouts by team
plt.figure(figsize=(10, 6))
sns.countplot(x="home_team", hue="events", data=strikeouts, order=teams)
plt.title("Count of Strikeouts by Team")
plt.xlabel("Home Team")
plt.ylabel("Count")
plt.show()

# Plot the count of hits by inning
plt.figure(figsize=(10, 6))
sns.countplot(x="inning", hue="events", data=hits)
plt.title("Count of Hits by Inning")
plt.xlabel("Inning")
plt.ylabel("Count")
plt.show()

# %%
# Box plot of launch speeds for different hits
sns.boxplot(data=hits, x='events', y='launch_speed')
plt.title("Launch Speed Distribution for Different Hit Types")
plt.show()

# %%
# the box plot shows the distribution of launch speed for each hit type, while the overlaid line plot
# (created with pointplot()) shows the mean launch speed for each hit type

# Prepare the data
hit_type_order = hits['events'].value_counts().index

# Create the box plot
sns.boxplot(data=hits, x='events', y='launch_speed',
            order=hit_type_order, color='lightblue', width=0.6)

# Create the point plot (line plot) using the mean value for each hit type
sns.pointplot(data=hits, x='events', y='launch_speed', order=hit_type_order,
              color='red', markers='o', join=True, ci=None, estimator=np.mean)

# Customize the plot
plt.title("Launch Speed Distribution by Hit Type with Mean Values")
plt.xlabel("Hit Type")
plt.ylabel("Launch Speed (mph)")

# Show the plot
plt.show()


# %%
# create pivot table of pitch outcomes by count
pivot_counts = df.pivot_table(
    values='pitch_type', index='count', columns='description', aggfunc=len)

# plot pitch outcomes by count
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(pivot_counts, cmap='Reds', annot=True,
            fmt='.0f', cbar=False, ax=ax)
ax.set_xlabel('Outcome')
ax.set_ylabel('Count')
ax.set_title('Pitch Outcomes by Count')
plt.show()


# %%
sns.displot(x='launch_angle', y='launch_speed',
            data=hits, hue='events', palette='Set1')
sns.displot(x='launch_angle', y='launch_speed', data=hits,
            hue='events', kind='kde', fill=True, levels=5, palette='Set1')
sns.displot(x='launch_angle', y='launch_speed', hue='bb_type',
            kind='hist', palette='Set1', data=hits)


# %% In[43]:
sns.displot(x='launch_angle', y='hit_distance_sc',
            data=hits, hue='events', palette='Set1')
sns.displot(x='launch_angle', y='hit_distance_sc', data=hits,
            hue='events', kind='kde', fill=True, levels=5, palette='Set1')
sns.displot(x='launch_angle', y='hit_distance_sc', hue='bb_type',
            kind='hist', palette='Set1', data=hits)


# %% In[75]:
sns.displot(x='pfx_x', y='pfx_z', data=not_in_play,
            hue='pitch_type', palette='Set1')
sns.displot(x='pfx_x', y='pfx_z', data=not_in_play, hue='pitch_type',
            kind='kde', fill=True, levels=5, palette='Set1')
sns.displot(x='pfx_x', y='pfx_z', hue='pitch_type',
            kind='hist', palette='Set1', data=not_in_play)

# %%leftys
sns.displot(x=float('release_spin_rate'), y='release_speed',
            data='lefty_st', hue='pitch_type', palette='Set1')
sns.displot(x=float('release_spin_rate'), y='release_speed', data='lefty_st',
            hue='pitch_type', kind='kde', fill=True, levels=5, palette='Set1')
sns.displot(x=float('release_spin_rate'), y='release_speed',
            hue='pitch_type', kind='hist', palette='Set1', data='lefty_st')


# %%stacking plots

x = hits['launch_angle']
y = hits['launch_speed']
h = hits['events']
f, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(x=x, y=y, hue=h, palette='Set1')
sns.histplot(x=x, y=y, hue=h, cmap="Set1")
sns.kdeplot(x=x, y=y, levels=5, hue=h, color="blue", linewidths=2)

# %%
xx = st_fastballs['release_spin_rate']
yy = st_fastballs['effective_speed']
hh = st_fastballs['zone']
f, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(x=xx, y=yy, hue=hh, palette='Set1')
sns.histplot(x=xx, y=yy, hue=hh)
sns.kdeplot(x=xx, y=yy, levels=5, hue=hh, color="blue", linewidths=2)

# %%
x = st_fastballs['plate_x']
y = st_fastballs['plate_z']
h = st_fastballs['zone']
f, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(x=x, y=y, hue=h, palette='Set1')
sns.histplot(x=x, y=y, hue=h)
sns.kdeplot(x=x, y=y, levels=5, hue=h, color="blue", linewidths=2)

# %%
x = strikeouts['plate_x']
y = strikeouts['plate_z']
h = strikeouts['pitch_name']
f, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(x=not_in_play['plate_x'], y=not_in_play['plate_z'],
                hue=not_in_play['pitch_name'], palette='Set1')
# sns.histplot(x=x, y=y,hue=h)
sns.displot(x=x, y=y, kind='kde', levels=5, hue=h, color="blue")

# %%sswarmplot
ax = sns.stripplot(x='events', y='hit_distance_sc',
                   hue='pitch_type', dodge=True, data=hits)
ax.set(ylabel='Hit Distance (ft)', xlabel="")

# %%
ax = sns.violinplot(x='events', y='launch_angle', dodge=True, data=hits)
ax.set(ylabel='Hit Distance (ft)', xlabel="")

# %%pointplot
ax = sns.catplot(x='events', y='launch_angle', kind='point', data=hits)
ax.set(ylabel='Launch angle (degrees)')


# %%columned catplot
ax = sns.catplot(x='bb_type', y='hit_distance_sc',
                 hue='events', col='stand', data=hits)
ax.set(ylabel='Hit Distance (ft)', xlabel="")

# %%
ax = sns.catplot(x='events', y='launch_angle',
                 hue='estimated_woba_using_speedangle', data=hits)
ax.set(ylabel='Hit Distance (ft)', xlabel="")

# %%
ax = sns.kdeplot(x='launch_angle', y='hit_distance_sc',
                 hue='events', col='stand', data=hits)
ax.set(ylabel='Hit Distance (ft)', xlabel="launch angle in degrees")

# %%
sns.stripplot(x='bb_type', y='launch_speed', hue='woba_value', data=hits)

# %%

sns.relplot(in_play['launch_angle'].astype(float),
            in_play['hit_distance_sc'].astype(float), cmap='gnuplot')
