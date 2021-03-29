import requests
import json
import os

url = 'https://rest.coinapi.io/v1/assets'
headers = {'X-CoinAPI-Key' : os.environ['COINAPI_KEY']}
response = requests.get(url, headers=headers)
with open('coin_assets.json', 'w+') as f:
    print('Writing coin_assets.json')
    json.dump(response.json(), f)
