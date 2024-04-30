import requests
import pandas as pd
url = "https://odds.p.rapidapi.com/v1/odds"

querystring = {"region": "us", "sport": "baseball_mlb",
               "oddsFormat": "american", "market": "h2h", "dateFormat": "iso"}

headers = {
    "X-RapidAPI-Host": "odds.p.rapidapi.com",
    "X-RapidAPI-Key": "b501e0bf75mshf881260fcf61406p1a7f13jsnbce859978dd1"
}

response = requests.request("GET", url, headers=headers, params=querystring)
# %% print odds
res = pd.read_json(response.text)
res = res['data']
for x in range(len(res)):
    home, away = res[x].get('teams')
    hodds, aodds = res[x].get('sites')[0].get('odds').get('h2h')


def odds_to_probability(odds):
    if odds < 0:
        prob = (-odds) / ((-odds) + 100)
    else:
        prob = 100 / (odds + 100)
    return prob


matchup_odds = {}

for x in range(len(res)):
    home, away = res[x].get('teams')
    hodds, aodds = res[x].get('sites')[0].get('odds').get('h2h')

    hprob = odds_to_probability(hodds)
    aprob = odds_to_probability(aodds)

    matchup = f"{away} vs {home}"
    output = f"Matchup: {away} has a {aprob*100:.2f}% chance to win against {home}. {home} has a {hprob*100:.2f}% chance to win against {away}."
    matchup_odds[matchup] = output

# You can now access the matchup odds from the dictionary
for matchup, output in matchup_odds.items():
    print(output)
