import pandas as pd

class CoinAssets(object):
    def __init__(self):
        # coin_assets.json retrieved from 
        self.df = pd.read_json('coin_assets.json', orient='records')
        self.df.set_index('asset_id', inplace=True, drop=True)
        self.df = self.df.query('type_is_crypto == 1 & volume_1mth_usd > 0')
        self.df = self.df.sort_values(by=['volume_1mth_usd'],ascending=False).head(100)

        self.coin_name_lookup = {}

    def get_asset_id(self, coin):
        asset_id = None
        if coin.upper() in self.df.index:
            asset_id = coin.upper()
        elif coin.lower() in self.coin_name_lookup:
            asset_id = self.coin_name_lookup[coin.lower()]
        else:
            found = self.df[self.df.name.str.lower() == coin.lower()]
            if not found.empty:
                asset_id = found.index.values[0]
                self.coin_name_lookup[coin.lower()] = asset_id

        return asset_id

    def get_list(self):
        assets = {}

        for asset in list(self.df.index.str.lower()):
            assets[str(asset).strip()] = 1

        for asset in list(self.df['name'].str.lower()):
            assets[str(asset).strip()] = 1

        return list(assets.keys())