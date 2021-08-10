from flask import Flask, render_template
import logging
import requests

from gevent import monkey
monkey.patch_all() # we need to patch very early

logging.basicConfig(format='[%(asctime)s] [%(filename)s:%(funcName)s:%(lineno)d] %(message)s', level=logging.DEBUG)

app = Flask(__name__, 
    template_folder="./application/templates",
    static_folder="./application/static",
    instance_relative_config=False)

import json

class XLNetRequest(object):
    def __init__(self):
        #self.url = 'http://localhost:8000'
        self.url = 'http://e77eff21f201.ngrok.io'

    def predict(self, body):
        data = {'body': body}
        x = requests.post(self.url + '/predict', json=data)
        return x.json()

    def summary(self):
        x = requests.get(self.url + '/summary')
        return json.loads(x.json())

@app.route("/")
def hello_world():
    xlnet_api = XLNetRequest()
    summary = xlnet_api.summary()
    return render_template('index.html', summary=summary)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)