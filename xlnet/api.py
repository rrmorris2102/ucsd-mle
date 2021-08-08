import sys
sys.path.append('../reddit')

import multiprocessing as mp
from multiprocessing import Lock
import threading, queue
import time
import pandas as pd
from cosmo_db import RedditDB

from flask import Flask
from flask_restful import reqparse, Resource, Api
from xlnet import XLNetSentiment, XLNetSentimentTrain

import logging

logging.basicConfig(format='[%(asctime)s] [%(filename)s:%(funcName)s:%(lineno)d] %(message)s', level=logging.INFO)

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('body', action='append')

model_file = './models/xlnet_model_batch48.bin'
xlnet = XLNetSentiment(model_file, batchsize=1, max_len=64)
summary_lock = Lock()

class Inference(Resource):
    def get(self):
        return 501

    def post(self):
        args = parser.parse_args()
        sentiment = []
        try:
            for body in args['body']:
                results = xlnet.predict(body)
                sentiment.append(results['sentiment'])
            return {'sentiment': sentiment}, 201
        except Exception as e:
            return 500

class Summary(Resource):
    def get(self):
        summary_lock.acquire()
        df = pd.read_csv('sentiment_summary.csv')
        summary_lock.release()
        result = df.to_json(orient='records')
        return result, 201

    def post(self):
        return 501

api.add_resource(Inference, '/predict')
api.add_resource(Summary, '/summary')

class SentimentSummaryWorker(threading.Thread):
    def __init__(self, requests, response):
        threading.Thread.__init__(self, daemon=True)

        self.requests = requests
        self.response = response

    def run(self):
        reddit_db = RedditDB()

        while True:
            request = self.requests.get()

            if request is None:
                self.requests.task_done()
                break

            coin = request['coin']
            coin_id = request['coin_id']

            logging.info('Coin {}'.format(coin))

            response = {'coin': coin, 'positive': 0, 'negative': 0, 'neutral': 0}

            comment_refs = reddit_db.coins.association.get(reddit_db.comments, coin_id)
            if comment_refs:
                for idx, ref in enumerate(comment_refs):
                    if idx % 100 == 0:
                        logging.info('[{}] Comment {}/{}'.format(coin, idx, len(comment_refs)))

                    comment = reddit_db.comments.get(columns=['sentiment'], db_id=ref['comments_id'])
                    if comment:
                        comment = comment[0]

                        if 'sentiment' in comment:
                            response[comment['sentiment']] += 1

            self.response.put(response)

            self.requests.task_done()

class MainProcess(mp.Process):
    def __init__(self, event, lock):
        self.event = event
        self.lock = lock
        mp.Process.__init__(self)

    def run(self):
        reddit_db = RedditDB()

        coins_df = reddit_db.coins.get_dataframe()
        top_coins = ['BTC', 'ETH', 'DOGE', 'ADA', 'ALGO', 'ETC', 'USDT', 'VET', 'XMR', 'XRP']
        coins_df = coins_df.loc[top_coins]

        requests = queue.Queue()
        responses = queue.Queue()
        threads = []

        max_workers = 10

        for idx in range(max_workers):
            x = SentimentSummaryWorker(requests, responses)
            threads.append(x)
            x.start()

        try:
            while True:
                tick = time.time()

                for coin in coins_df.index:
                    coin_info = coins_df.loc[coin]

                    request = {'coin': coin, 'coin_id': coin_info['id']}
                    requests.put(request)

                requests.join()

                data = {
                    'coin': [],
                    'positive': [],
                    'negative': [],
                    'neutral': []
                }

                while not responses.empty():
                    response = responses.get()

                    data['coin'].append(response['coin'])
                    data['positive'].append(response['positive'])
                    data['negative'].append(response['negative'])
                    data['neutral'].append(response['neutral'])

                comments_df = pd.DataFrame.from_dict(data)
                comments_df = comments_df.set_index('coin')

                self.lock.acquire()
                comments_df.to_csv('sentiment_summary.csv')
                self.lock.release()

                tock = time.time()

                logging.info('Elapsed time {} seconds'.format(tock - tick))

                self.event.wait(3600)

        except KeyboardInterrupt:
            logging.debug('interrupted!')

            for x in threads:
                requests.put(None)

        except Exception as e:
            logging.error(e)

        for x in threads:
            requests.put(None)

if __name__ == '__main__':
    main = MainProcess(mp.Event(), summary_lock)
    main.start(),

    app.run(debug=False, port=8000)

    main.join()
