# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 17:50:40 2023

@author: rhoffpauir
"""

# Import required libraries
import streamlit as st
import pybaseball
from pybaseball import playerid_lookup, statcast_batter, batting_stats
import plotly.express as px
import pandas as pd

# Streamlit App
st.title('Sabermetric Dashboard')

# Search bar for player names
player_fname = st.text_input("Search Player First Name:", "")
player_lname = st.text_input("Search Player Last Name:", "")

# Fetch player ID
try:
    player_ids = playerid_lookup(player_lname, player_fname)
    player_id = pd.DataFrame(player_ids['key_bbref'])
except:
    st.write("Player not found. Please enter a valid player name.")
    st.stop()

# Fetch basic player info
batting = pybaseball.batting()
batting = batting[batting['yearID'] >= 1950]
basic_stats = batting.loc[batting['playerID'].isin(player_id['key_bbref'])]
basic_stats['1B'] = basic_stats['H'].sum()-(basic_stats['2B'].sum() +
                                            basic_stats['3B'].sum()+basic_stats['HR'].sum())
basic_stats['TB'] = basic_stats['1B'].sum()+basic_stats['2B'].sum()*2 + \
    basic_stats['3B'].sum()*3+basic_stats['HR'].sum()*4
if basic_stats.empty:
    st.write("Player stats for the current season not found.")
    st.stop()

# Display basic info
st.write(f"ID: {basic_stats['playerID'].values[0]}")
st.write(
    f"Games/At-Bats: {basic_stats['G'].sum()}/{basic_stats['AB'].sum()}")
st.write(f"Hits: {basic_stats['H'].sum()}")
st.write(f"HomeRuns: {basic_stats['HR'].sum()}")
st.write(f"Total Bases: {basic_stats['TB'].values[0]}")
st.write(f"Team: {basic_stats['teamID'].values[0]}")

# Fetch statcast data
statcast_data = statcast_batter('2022-01-01', '2022-12-31', player_id)

# Platoon Splits
platoon_data = statcast_data.groupby(
    'stand').mean()[['launch_speed', 'launch_angle']]
fig1 = px.bar(platoon_data.reset_index(), x='stand', y=[
              'launch_speed', 'launch_angle'], title='Platoon Splits')
st.plotly_chart(fig1)

# GB%, FB%, GB/FB%
batted_ball_data = statcast_data['bb_type'].value_counts(normalize=True) * 100
fig2 = px.bar(batted_ball_data.reset_index(), x='index',
              y='bb_type', title='Batted Ball Percentages')
st.plotly_chart(fig2)

# Expected vs Actual Stats
expected_stats = statcast_data[[
    'estimated_ba_using_speedangle', 'estimated_woba_using_speedangle']].mean()
actual_stats = statcast_data[['batting_avg', 'woba_value']].mean()
comparison_data = pd.DataFrame(
    {'Expected': expected_stats, 'Actual': actual_stats})
fig3 = px.bar(comparison_data.reset_index(), x='index', y=[
              'Expected', 'Actual'], title='Expected vs Actual Stats')
st.plotly_chart(fig3)
