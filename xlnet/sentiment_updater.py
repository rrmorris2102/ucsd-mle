import sys
sys.path.append('../reddit')

import os
import multiprocessing as mp
import pandas as pd
from nltk.tokenize import sent_tokenize
from cosmo_db import RedditDB
import time
import logging
from xlnet_api import XLNetRequest

logging.basicConfig(format='[%(asctime)s] [%(filename)s:%(funcName)s:%(lineno)d] %(message)s', level=logging.INFO)

class CoinProcess(mp.Process):
    def __init__(self, requests):
        mp.Process.__init__(self)
        self.requests = requests

    def run(self):
        reddit_db = RedditDB()
        xlnet_api = XLNetRequest()

        while True:
            request = self.requests.get()

            if request is None:
                self.requests.task_done()
                break

            coin = request['coin']
            coin_id = request['coin_id']

            logging.info('Coin {}'.format(coin))

            data = {
                'coin_id': [],
                'comment_id': [],
                'body': [],
                'sentiment': []
            }

            sentiment_updated = 0

            try:
                comment_refs = reddit_db.coins.association.get(reddit_db.comments, coin_id)
                if comment_refs:
                    for idx, ref in enumerate(comment_refs):
                        if idx % 100 == 0:
                            logging.info('[{}] Comment {}/{}'.format(coin, idx, len(comment_refs)))
                        comment = reddit_db.comments.get(db_id=ref['comments_id'])
                        if comment:
                            comment = comment[0]
                            data['coin_id'].append(coin)
                            data['comment_id'].append(comment['comment_id'])
                            data['body'].append(comment['body'])

                            if 'sentiment' in comment:
                                data['sentiment'].append(comment['sentiment'])
                            elif comment['body'].strip() == '':
                                data['sentiment'].append(None)
                            else:
                                sentences = sent_tokenize(comment['body'])
                                sentiment = {'positive': 0, 'negative': 0, 'neutral': 0}
                                results = xlnet_api.predict(sentences)
                                for result in results['sentiment']:
                                    sentiment[result] += 1

                                if sentiment['positive'] > sentiment['negative']:
                                    data['sentiment'].append('positive')
                                elif sentiment['negative'] > sentiment['positive']:
                                    data['sentiment'].append('negative')
                                else:
                                    data['sentiment'].append('neutral')

                                comment['sentiment'] = data['sentiment'][-1]
                                reddit_db.comments.add(comment)
                                sentiment_updated += 1
            except Exception as e:
                logging.error(e)

            logging.info('[{}] {} sentiments updated'.format(coin, sentiment_updated))

            #comments_df = pd.DataFrame.from_dict(data)
            #comments_df.to_csv('coins_comments.csv')

            self.requests.task_done()

def xlnet_predict():
    reddit_db = RedditDB()

    coins_df = reddit_db.coins.get_dataframe()

    # Skip coin names that are common dictionary words
    skip_coins = ['ONE', 'MOON', 'LONG', 'BEAR', 'BULL', 'LINK', 'CASH', 'DOT', 'HOT', 'SUN', 'POT', 'BOX', 'LEND', 'DASH', 'MASK', 'WHITE']
    coins_df = coins_df.drop(index=skip_coins)

    count = []
    for coin in coins_df.index:
        coin_info = coins_df.loc[coin]

        comments = reddit_db.coins.association.get(reddit_db.comments, coin_info['id'])
        comments_count = 0
        if comments:
            comments_count = len(comments)

        count.append(comments_count)

        if comments_count > 0:
            logging.info('{}\t{}'.format(coin, comments_count))

    coins_df['count'] = count
    coins_df = coins_df.sort_values(by=['count'], ascending=False)
    coins_df[['name', 'volume_1mth_usd', 'count']].to_csv('top_coins.csv')

    coins_df = coins_df.head(10)

    data = {
        'coin_id': [],
        'comment_id': [],
        'body': [],
        'sentiment': []
    }

    processes = []
    requests = mp.JoinableQueue()

    for idx in range(10):
        x = CoinProcess(requests)
        processes.append(x)
        x.start()

    sleep_delay = int(os.getenv('REDDIT_SLEEP_DELAY', 300))

    try:
        while True:
            logging.info('Wake up')

            tick = time.time()

            for coin in coins_df.index:
                coin_info = coins_df.loc[coin]
                requests.put({'coin': coin, 'coin_id': coin_info['id']})

            requests.join()

            tock = time.time()

            logging.info('Elapsed time {} seconds'.format(tock - tick))

            logging.info('Sleeping for {} seconds'.format(sleep_delay))

            event.wait(sleep_delay)

    except KeyboardInterrupt:
        logging.debug('interrupted!')

        for x in processes:
            requests.put(None)

    except Exception as e:
        logging.error(e)

    for x in processes:
        x.join()

class MainProcess(mp.Process):
    def __init__(self, event):
        self.event = event
        mp.Process.__init__(self)

    def run(self):
        xlnet_predict()

if __name__ == '__main__':
    event = mp.Event()
    main = MainProcess(event)
    main.start()
    main.join()

