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
    print(f"{away}: {aodds} @ {home}: {hodds}")
