import os
import requests

class XLNetRequest(object):
    def __init__(self):
        self.url = os.getenv('XLNET_URL', 'http://localhost:8000')

    def predict(self, body):
        data = {'body': body}
        x = requests.post(self.url + '/predict', json=data)
        return x.json()

    def summary(self):
        x = requests.get(self.url + '/summary')
        return x.json()