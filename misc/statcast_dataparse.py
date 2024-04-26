# -*- coding: utf-8 -*-
"""
Created on Thu May 25 13:24:59 2023

@author: rhoffpauir
"""

# %%
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

# Create a SQLite3 database and commit the DataFrames as tables
# conn = sqlite3.connect('.\data\statcast_events.db')
# hits.to_sql(name='hits', con=conn)
# strikes.to_sql(name='strikes', con=conn)
# balls.to_sql(name='balls', con=conn)
# lefty_st.to_sql(name='lefty_st', con=conn)
# righty_st.to_sql(name='righty_st', con=conn)
# st_fastballs.to_sql(name='st_fastballs', con=conn)
# st_offspeed.to_sql(name='st_offspeed', con=conn)
# strikeouts.to_sql(name='strikeouts', con=conn)
# conn.close()
