import os
import multiprocessing as mp
import threading, queue
from time import sleep
from coin import CoinAssets
from crypto_sentiment import WordFrequency
from cosmo_db import RedditDB
import pandas as pd
from datetime import datetime, timezone
import time
from reddit import RedditApi
import logging

logging.basicConfig(format='[%(asctime)s] [%(filename)s:%(funcName)s:%(lineno)d] %(message)s', level=logging.INFO)

class RedditContext():
    def __init__(self, lock):
        self.reddit = RedditApi()
        self.coin_assets = CoinAssets()
        self.coins = self.coin_assets.get_list()
        self.wf = WordFrequency(filter=self.coins)
        self.refs = 0
        self.lock = lock

    def attach(self):
        with self.lock:
            self.refs += 1

            if self.refs == 1:
                self.reddit_db = RedditDB()
                self.coins_db = self.reddit_db.coins.get_dataframe()

    def detach(self):
        with self.lock:
            if self.refs > 0:
                self.refs -= 1

            if self.refs == 0 and not self.reddit_db is None:
                del self.reddit_db
                self.reddit_db = None
                self.coins_db = None

class RedditArticleWorker(threading.Thread):
    def __init__(self, context, subreddit_name, subreddit_id, last_article_seconds, requests, results):
        threading.Thread.__init__(self, daemon=True)

        self.context = context
        self.subreddit_name = subreddit_name
        self.subreddit_id = subreddit_id
        self.last_article_seconds = last_article_seconds
        self.requests = requests
        self.results = results

    def run(self):
        reddit = self.context.reddit
        coin_assets = self.context.coin_assets
        coins = self.context.coins
        last_article_seconds = self.last_article_seconds
        wf = self.context.wf

        group = self.subreddit_name

        while True:
            request = self.requests.get()

            if request is None:
                self.requests.task_done()
                break

            article_id = request['article_id']
            article = request['article']

            logging.debug('Worker {} getting comments for article {}'.format(threading.get_ident(), article_id))

            retries = 3
            while retries > 0:
                try:
                    self.context.attach()
                    reddit_db = self.context.reddit_db
                    coins_db = self.context.coins_db

                    add_comments = []

                    last_comment_seconds = None
                    last_comment = reddit_db.comments.get_last(filter='article_id = "{}"'.format(article_id))
                    if last_comment:
                        last_comment = last_comment[0]
                        last_comment_seconds = last_comment['created_utc']

                    comments = reddit.get_comments(group, article_id)
                    if comments:
                        logging.debug('Fetched {} comments'.format(comments.count()))

                        for comment_id, comment in comments.df.iterrows():
                            check_fields = ['created_utc', 'author_fullname', 'author', 'body']

                            fields_valid = True
                            for field in check_fields:
                                if pd.isna(comment[field]):
                                    fields_valid = False
                                    break

                            if fields_valid == False:
                                continue

                            wf.clear()
                            wf.add_words(comment['body'], comment_id)

                            created_utc_dt = pd.to_datetime(comment['created_utc'])
                            created_utc_seconds = int(created_utc_dt.replace(tzinfo=timezone.utc).timestamp())

                            if wf.count() > 0 and (not last_comment_seconds or (last_comment_seconds and created_utc_seconds > last_comment_seconds)):
                                logging.debug('{} coin mentions in comment {}'.format(wf.count(), comment_id))
                                add_comments.append(
                                    {
                                        'comment_id': comment_id,
                                        'created_utc': created_utc_seconds,
                                        'word_list': wf.word_list
                                    }
                                )

                        wf.clear()
                        wf.add_words(article['title'], article_id)
                        wf.add_words(article['selftext'], article_id)

                        created_utc_dt = pd.to_datetime(article['created_utc'])
                        created_utc_seconds = int(created_utc_dt.replace(tzinfo=timezone.utc).timestamp())

                        if (not last_article_seconds or (last_article_seconds and created_utc_seconds > last_article_seconds)) and (wf.count() > 0 or len(add_comments) > 0):
                            reddit_db.authors.add(
                                {
                                    'author_id': article['author_fullname'],
                                    'name': article['author']
                                })


                            reddit_db.articles.add(
                                {
                                    'article_id': article_id,
                                    'subreddit_id': self.subreddit_id,
                                    'created_utc': created_utc_seconds,
                                    'selftext': article['selftext'],
                                    'title': article['title'],
                                    'permalink': article['permalink'],
                                    'upvote_ratio': article['upvote_ratio'],
                                    'ups': int(article['ups']),
                                    'downs': int(article['downs']),
                                    'score': int(article['score']),
                                })

                            article_db = reddit_db.articles.get(id=article_id)
                            if article_db:
                                article_db = article_db[0]
                                for word in wf.word_list:
                                    coin_id = coin_assets.get_asset_id(word)

                                    if coin_id:
                                        reddit_db.coins.association.set(
                                            reddit_db.articles,
                                            coins_db.loc[coin_id]['id'],
                                            article_db['id']
                                        )
                                        logging.debug('coin_id {}'.format(coin_id))

                            for add_comment in add_comments:
                                comment_id = add_comment['comment_id']
                                comment = comments.df.loc[comment_id]

                                reddit_db.authors.add(
                                    {
                                        'author_id': comment['author_fullname'],
                                        'name': comment['author']
                                    })

                                reddit_db.comments.add(
                                        {
                                            'comment_id': comment_id,
                                            'body': comment['body'],
                                            'parent_id': comment['parent_id'],
                                            'article_id': article_id,
                                            'created_utc': add_comment['created_utc'],
                                            'permalink': comment['permalink'],
                                            'ups': int(comment['ups'])
                                        })

                                comment_db = reddit_db.comments.get(id=comment_id)
                                if comment_db:
                                    comment_db = comment_db[0]

                                    word_list = add_comment['word_list']
                                    for word in word_list:
                                        coin_id = coin_assets.get_asset_id(word)

                                        if coin_id:
                                            reddit_db.coins.association.set(
                                                reddit_db.comments,
                                                coins_db.loc[coin_id]['id'],
                                                comment_db['id']
                                            )
                                            logging.debug('coin_comment {} {}'.format(coins_db.loc[coin_id]['id'], comment_db['id']))

                    self.context.detach()
                    retries = 0

                except Exception as e:
                    logging.error(e)
                    time.sleep(60)
                    retries -= 1
                    self.context.detach()

            self.requests.task_done()

class RedditSubredditProcess(mp.Process):
    def __init__(self, requests):
        mp.Process.__init__(self)
        self.requests = requests
        lock = threading.Lock()
        self.reddit_context = RedditContext(lock)

    def run(self):
        reddit = self.reddit_context.reddit
        coins = self.reddit_context.coins

        while True:
            request = self.requests.get()

            if request is None:
                self.requests.task_done()
                break

            self.reddit_context.attach()
            reddit_db = self.reddit_context.reddit_db

            group = request['subreddit']
            logging.info('Update subreddit started {}'.format(group))
            articles = reddit.get_articles(group)

            # Get subreddit info from first record
            subreddit = articles.df.iloc[0]
            reddit_db.subreddits.add(
                {
                    'subreddit_id': subreddit.subreddit_id,
                    'name': subreddit.subreddit,
                    'subscribers': int(subreddit.subreddit_subscribers)
                })

            logging.debug('{} subreddits'.format(reddit_db.subreddits.count()))

            wf = WordFrequency(filter=coins)

            last_article_seconds = None
            last_article = reddit_db.articles.get_last(filter='subreddit_id = "{}"'.format(subreddit.subreddit_id))
            if last_article:
                last_article = last_article[0]
                last_article_seconds = last_article['created_utc']

            # Thread pool
            threads = []
            max_workers = 8
            requests = queue.Queue()
            results = queue.Queue()

            for idx in range(max_workers):
                x = RedditArticleWorker(self.reddit_context, group, subreddit.subreddit_id, last_article_seconds, requests, results)
                threads.append(x)
                x.start()

            # Add authors and articles
            for article_id, article in articles.df.iterrows():
                request = {'article_id': article_id, 'article': article}
                requests.put(request)

            for idx in range(max_workers):
                requests.put(None)

            requests.join()

            for x in threads:
                x.join()

            logging.info('Update subreddit finished {}'.format(group))
            self.reddit_context.detach()
            self.requests.task_done()

def reddit_sync(event):
    reddit = RedditApi()

    coin_assets = CoinAssets()
    coins = coin_assets.get_list()

    logging.debug('Found {} coin names'.format(len(coins)))

    subreddits = [
        '/r/cryptomarkets',
        '/r/cryptocurrency',
        '/r/cryptocurrencies',
        '/r/cryptomoonshots',
        '/r/satoshistreetbets'
    ]

    reddit_db = RedditDB()

    # Populate coins
    if reddit_db.coins.count() != len(coin_assets.df):
        for coin_id, coin in coin_assets.df.iterrows():
            coin_name = coin_id if pd.isna(coin['name']) else coin['name']
            volume_1hrs_usd = 0 if pd.isna(coin['volume_1hrs_usd']) else coin['volume_1hrs_usd']
            volume_1day_usd = 0 if pd.isna(coin['volume_1day_usd']) else coin['volume_1day_usd']
            volume_1mth_usd = 0 if pd.isna(coin['volume_1mth_usd']) else coin['volume_1mth_usd']
            price_usd = 0 if pd.isna(coin['price_usd']) else coin['price_usd']
            id_icon = 0 if pd.isna(coin['id_icon']) else coin['id_icon']

            reddit_db.coins.add(
                {
                    'coin_id': coin_id,
                    'name': coin_name,
                    'volume_1hrs_usd': volume_1mth_usd,
                    'volume_1day_usd': volume_1day_usd,
                    'volume_1mth_usd': volume_1mth_usd,
                    'price_usd': price_usd,
                    'id_icon': id_icon,
                    'updated_utc': datetime.utcnow().isoformat()
                }
            )

    coins_db = reddit_db.coins.get_dataframe()

    # Skip coin names that are common dictionary words
    skip_coins = ['ONE', 'MOON', 'LONG', 'BEAR', 'LINK', 'CASH', 'DOT', 'HOT', 'SUN', 'POT', 'BOX', 'LEND', 'DASH', 'MASK']
    coins_db = coins_db.drop(index=skip_coins)

    sleep_delay = int(os.getenv('REDDIT_SLEEP_DELAY', 300))

    processes = []
    requests = mp.JoinableQueue()

    for group in subreddits:
        x = RedditSubredditProcess(requests)
        processes.append(x)
        x.start()

    try:
        while True:
            logging.info('Wake up')

            try:
                del reddit_db
                reddit_db = RedditDB()

                num_authors = reddit_db.authors.count()
                num_articles = reddit_db.articles.count()
                num_comments = reddit_db.comments.count()

                tick = time.time()

                for group in subreddits:
                    requests.put({'subreddit': group})

                requests.join()

                tock = time.time()

                logging.info('Elapsed time {} seconds'.format(tock - tick))

                count = reddit_db.authors.count()
                logging.info('{} authors added'.format(count - num_authors))
                logging.info('{} total authors'.format(count))

                count = reddit_db.articles.count()
                logging.info('{} articles added'.format(count - num_articles))
                logging.info('{} total articles'.format(count))

                count = reddit_db.comments.count()
                logging.info('{} comments added'.format(count - num_comments))
                logging.info('{} total comments'.format(count))

                logging.info('Sleeping for {} seconds'.format(sleep_delay))
            except Exception as e:
                logging.error(e)

            event.wait(sleep_delay)
    except KeyboardInterrupt:
        logging.debug('interrupted!')

    for x in processes:
        x.join()

class MainProcess(mp.Process):
    def __init__(self, event):
        self.event = event
        mp.Process.__init__(self)

    def run(self):
        reddit_sync(event)

if __name__ == '__main__':
    event = mp.Event()
    main = MainProcess(event)
    main.start()
    main.join()
