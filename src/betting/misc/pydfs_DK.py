# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 04:15:31 2021

@author: robin
"""
from pydfs_lineup_optimizer import get_optimizer, Site, Sport, CSVLineupExporter


optimizer = get_optimizer(Site.DRAFTKINGS, Sport.BASKETBALL)
optimizer.load_players_from_csv("Downloads/DKSalaries(2).csv")
optimizer.set_max_repeating_players(3)
for lineup in optimizer.optimize(n=20):
    print(lineup)
    print(lineup.players)  # list of players
    print(lineup.fantasy_points_projection)
    print(lineup.salary_costs)

optimizer.export("DKNBA11_05.csv")
