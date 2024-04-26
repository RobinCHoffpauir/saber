# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 17:06:32 2021

@author: robin
"""
import pybaseball as pyb
import pandas as pd
import random

# Example of players' batting averages
players = pyb.batting_stats_bref(2023)

# Function to simulate an at-bat
def at_bat(player):
    hit_probability = players['BA']
    return random.random() < hit_probability

# Simulate an inning
def simulate_inning():
    outs = 0
    bases = [False, False, False]  # First, second, third bases
    runs = 0
    while outs < 3:
        for player in players:
            if at_bat(player):
                # This is a very simplified handling of base runners
                if bases[2]:  # If third base is occupied
                    runs += 1
                    bases[2] = False
                # Move runners
                bases = [True] + bases[:-1]
            else:
                outs += 1
            if outs >= 3:
                break
    return runs

# Simulate a game
def simulate_game():
    total_runs = 0
    for _ in range(9):  # 9 innings
        total_runs += simulate_inning()
    return total_runs

# Main simulation
n_games = 1000
total_runs = sum(simulate_game() for _ in range(n_games))
average_runs = total_runs / n_games
print(f'Average runs per game: {average_runs:.2f}')
