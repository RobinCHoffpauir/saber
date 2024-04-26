# %% Wins to date, losses to date and win percentage for each team


# Gets wins-to-date, losses-to-date, and win_percent-to-date for each team
brewers['Wins'] = np.where(brewers['W/L'] == 'W', 1,
                           (np.where(brewers['W/L'] == 'W-wo', 1, 0))).cumsum()
brewers['Losses'] = np.where(
    brewers['W/L'] == 'L', 1, (np.where(brewers['W/L'] == 'L-wo', 1, 0))).cumsum()
brewers['Win_Percentage'] = brewers['Wins'] / \
    (brewers['Wins'] + brewers['Losses'])

giants['Wins'] = np.where(giants['W/L'] == 'W', 1,
                          (np.where(giants['W/L'] == 'W-wo', 1, 0))).cumsum()
giants['Losses'] = np.where(giants['W/L'] == 'L', 1,
                            (np.where(giants['W/L'] == 'L-wo', 1, 0))).cumsum()
giants['Win_Percentage'] = giants['Wins'] / (giants['Wins'] + giants['Losses'])

rays['Wins'] = np.where(rays['W/L'] == 'W', 1,
                        (np.where(rays['W/L'] == 'W-wo', 1, 0))).cumsum()
rays['Losses'] = np.where(rays['W/L'] == 'L', 1,
                          (np.where(rays['W/L'] == 'L-wo', 1, 0))).cumsum()
rays['Win_Percentage'] = rays['Wins'] / (rays['Wins'] + rays['Losses'])

white_sox['Wins'] = np.where(
    white_sox['W/L'] == 'W', 1, (np.where(white_sox['W/L'] == 'W-wo', 1, 0))).cumsum()
white_sox['Losses'] = np.where(
    white_sox['W/L'] == 'L', 1, (np.where(white_sox['W/L'] == 'L-wo', 1, 0))).cumsum()
white_sox['Win_Percentage'] = white_sox['Wins'] / \
    (white_sox['Wins'] + white_sox['Losses'])

dodgers['Wins'] = np.where(dodgers['W/L'] == 'W', 1,
                           (np.where(dodgers['W/L'] == 'W-wo', 1, 0))).cumsum()
dodgers['Losses'] = np.where(
    dodgers['W/L'] == 'L', 1, (np.where(dodgers['W/L'] == 'L-wo', 1, 0))).cumsum()
dodgers['Win_Percentage'] = dodgers['Wins'] / \
    (dodgers['Wins'] + dodgers['Losses'])

astros['Wins'] = np.where(astros['W/L'] == 'W', 1,
                          (np.where(astros['W/L'] == 'W-wo', 1, 0))).cumsum()
astros['Losses'] = np.where(astros['W/L'] == 'L', 1,
                            (np.where(astros['W/L'] == 'L-wo', 1, 0))).cumsum()
astros['Win_Percentage'] = astros['Wins'] / (astros['Wins'] + astros['Losses'])

red_sox['Wins'] = np.where(red_sox['W/L'] == 'W', 1,
                           (np.where(red_sox['W/L'] == 'W-wo', 1, 0))).cumsum()
red_sox['Losses'] = np.where(
    red_sox['W/L'] == 'L', 1, (np.where(red_sox['W/L'] == 'L-wo', 1, 0))).cumsum()
red_sox['Win_Percentage'] = red_sox['Wins'] / \
    (red_sox['Wins'] + red_sox['Losses'])

d_backs['Wins'] = np.where(d_backs['W/L'] == 'W', 1,
                           (np.where(d_backs['W/L'] == 'W-wo', 1, 0))).cumsum()
d_backs['Losses'] = np.where(
    d_backs['W/L'] == 'L', 1, (np.where(d_backs['W/L'] == 'L-wo', 1, 0))).cumsum()
d_backs['Win_Percentage'] = d_backs['Wins'] / \
    (d_backs['Wins'] + d_backs['Losses'])

# %%


# Graph Wins Comparison #
plt.rcParams["figure.figsize"] = (10, 8)

plt.plot(brewers['Wins'], label='Brewers', c='navy')
plt.plot(dodgers['Wins'], label='Dodgers', c='blue')
plt.plot(giants['Wins'], label='Giants', c='orange')
plt.plot(rays['Wins'], label='Rays', c='yellow')
plt.plot(astros['Wins'], label='Astros', c='green')
plt.plot(red_sox['Wins'], label='Red Sox', c='pink')
plt.plot(white_sox['Wins'], label='White Sox', c='black')
plt.plot(d_backs['Wins'], label='Diamondbacks', c='purple')

plt.xticks(np.arange(0, len(brewers.index), step=10))
plt.xlabel('Game')
plt.ylabel('Wins')

plt.legend(loc='lower right')

plt.title(f"MLB Top Teams {year} Wins Comparison")

# %%

# Graph Win Percentage Comparison #
plt.rcParams["figure.figsize"] = (8, 6)

plt.plot(brewers['Win_Percentage'], label='Brewers', c='navy')
plt.plot(dodgers['Win_Percentage'], label='Dodgers', c='blue')
plt.plot(giants['Win_Percentage'], label='Giants', c='orange')
plt.plot(rays['Win_Percentage'], label='Rays', c='yellow')
plt.plot(astros['Win_Percentage'], label='Astros', c='green')
plt.plot(red_sox['Win_Percentage'], label='Red Sox', c='pink')
plt.plot(white_sox['Win_Percentage'], label='White Sox', c='black')

plt.xticks(np.arange(0, len(brewers.index), step=10))
plt.xlabel('Game')
plt.ylabel('Win Percentage')

plt.legend(loc='lower right')

plt.title("MLB Top Teams 2021 Win Percentage Comparison ({})".format(year))

# %%
# Get total runs-to-date and total runs_allowed-to-date for each team
brewers['Total_Runs'] = brewers['R'].cumsum()
brewers['Total_RA'] = brewers['RA'].cumsum()

dodgers['Total_Runs'] = dodgers['R'].cumsum()
dodgers['Total_RA'] = dodgers['RA'].cumsum()

white_sox['Total_Runs'] = white_sox['R'].cumsum()
white_sox['Total_RA'] = white_sox['RA'].cumsum()

giants['Total_Runs'] = giants['R'].cumsum()
giants['Total_RA'] = giants['RA'].cumsum()

rays['Total_Runs'] = rays['R'].cumsum()
rays['Total_RA'] = rays['RA'].cumsum()

astros['Total_Runs'] = astros['R'].cumsum()
astros['Total_RA'] = astros['RA'].cumsum()

red_sox['Total_Runs'] = red_sox['R'].cumsum()
red_sox['Total_RA'] = red_sox['RA'].cumsum()


# %%

# Graph Runs and Runs Allowed Comparisons #
plt.rcParams["figure.figsize"] = (16, 8)

plt.subplot(1, 2, 1)
plt.plot(brewers['Total_Runs'], label='Brewers', c='navy')
plt.plot(dodgers['Total_Runs'], label='Dodgers', c='#b30000')
plt.plot(rays['Total_Runs'], label='Rays', c='yellow')
plt.plot(astros['Total_Runs'], label='Astros', c='green')
plt.plot(white_sox['Total_Runs'], label='White Sox', c='red')
plt.plot(red_sox['Total_Runs'], label='Red Sox', c='pink')
plt.plot(giants['Total_Runs'], label='Giants', c='orange')


plt.xticks(np.arange(0, len(brewers.index), step=10))
plt.xlabel('Game')
plt.ylabel('Runs')

plt.title("MLB Best Team Runs Comparison ({})".format(year))
plt.legend(loc='lower right')

# %%