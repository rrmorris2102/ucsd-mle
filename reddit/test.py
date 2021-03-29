import requests
import requests.auth
import os
import string
from datetime import datetime
import pytz
import pandas as pd
import nltk

class RedditArticles(object):
    def __init__(self, response):
        article_summary = {
            'subreddit': [],
            'title': [],
            'name': [],
            'author_fullname': [],
            'created_utc': [],
            'selftext': [],
            'upvote_ratio': [],
            'ups': [],
            'downs': [],
            'score': []
        }

        for post in response['data']['children']:
            for column in article_summary.keys():
                if column == 'title' or column == 'selftext':
                    #article_summary[column].append(post['data'][column].encode("utf-8"))
                    article_summary[column].append(post['data'][column])
                elif column == 'created_utc':
                    dt_utc = datetime.fromtimestamp(post['data'][column])
                    dt_utc = dt_utc.astimezone(pytz.utc)
                    article_summary[column].append(dt_utc)
                else:
                    article_summary[column].append(post['data'][column])

        df = pd.DataFrame.from_dict(article_summary)
        df = df.set_index('name')        
        #for key in article_summary.keys():
            #df[key] = pd.Series(article_summary[key])

        self.df = df

    def count(self):
        return len(self.df)
    
class RedditComments(object):
    def __init__(self, response):
        comment_summary = {
            'name': [],
            'body': [],
            'author_fullname': [],
            'created_utc': []
        }
        self.df = None

        def collect_comments(comment):
            for column in comment_summary.keys():
                #print('comment {}'.format(comment['data'].keys()))
                if column in comment['data']:
                    if column == 'created_utc':
                        dt_utc = datetime.fromtimestamp(comment['data'][column])
                        dt_utc = dt_utc.astimezone(pytz.utc)
                        comment_summary[column].append(dt_utc)
                    else:
                        #comment_summary[column].append(comment['data'][column].encode("utf-8"))
                        comment_summary[column].append(comment['data'][column])

            if 'replies' in comment['data'] and comment['data']['replies'] != '':
                for reply in comment['data']['replies']['data']['children']:
                    collect_comments(reply)

        # [0] has the title
        # [1] has the comments and replies
        for comment in response[1]['data']['children']:
            collect_comments(comment)

        df = pd.DataFrame()
        for key in comment_summary.keys():
            df[key] = pd.Series(comment_summary[key])
        df = df.set_index('name')

        self.df = df

    def count(self):
        if self.df is None:
            return 0

        return len(self.df)

class RedditApi(object):
    # https://www.reddit.com/dev/api

    def __init__(self):
        self.user_agent = 'HammerTrade/0.1 by rrmorris'
        self.access_token = None

        if 'REDDIT_AUTH' in os.environ:
            credentials = os.environ['REDDIT_AUTH']

            client_auth = requests.auth.HTTPBasicAuth('LBosXQAthwrVIg', 'q9HTClR6Y37TyrbAKdVLuakOI25_bw')
            post_data = {"grant_type": "password", "username": "rrmorris", "password": credentials}
            headers = {"User-Agent": self.user_agent}
            response = requests.post("https://www.reddit.com/api/v1/access_token", auth=client_auth, data=post_data, headers=headers)
            response = response.json()
            print(response)
            if not 'access_token' in response:
                raise Exception('Unable to retrieve access token')

            self.access_token = response['access_token']
    
    def about_me(self):
        if self.access_token:
            headers = {"Authorization": 'bearer {}'.format(self.access_token), "User-Agent": self.user_agent}
            response = requests.get("https://oauth.reddit.com/api/v1/me", headers=headers)
            return response.json()

        return None

    def get_articles(self, subreddit):
       
        url = 'https://www.reddit.com/{}/top/.json?limit=500'.format(subreddit)
        print('url {}'.format(url))
        headers = {"User-Agent": self.user_agent}
        response = requests.get(url, headers=headers).json()

        if 'error' in response:
            raise Exception('Error {}: {}'.format(response['error'], response['message']))

        return RedditArticles(response)
    
    def get_comments(self, subreddit, article):
        if article.startswith('t3_'):
            article = article[3:]
        url = 'https://www.reddit.com/{}/comments/{}/.json'.format(subreddit, article)
        print('url {}'.format(url))
        headers = {"User-Agent": self.user_agent}
        response = requests.get(url, headers=headers).json()

        if 'error' in response:
            raise Exception('Error {}: {}'.format(response['error'], response['message']))

        return RedditComments(response)

class WordFrequency(object):
    def __init__(self, filter=None):
        self.stop_words = ['b', 'i', 'are', 'as', 'but', 'we', 'an', 'by', 'has', 'or', 'was', 'that', 'me', 'like', 'have', 'if', 'so', 'your', 'the', 'and', 'for', 'on', 'to', 'I', 'a', 'of', 'with', 'is', 'my', 'you', 'be', 'in', 'this', 'at', 'it', 'from', 'after', 'their', '-']
        self.word_summary = {}
        self.df = None
        self.table = str.maketrans(dict.fromkeys(string.punctuation))
        self.filter = filter
    
    def add_words(self, words, ref):
        # https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string        

        for word in words:
            word = word.strip().lower().translate(self.table)

            if word == '':
                continue

            if not word in self.stop_words:
                if not self.filter or (self.filter and word in self.filter):
                    if not word in self.word_summary:
                        self.word_summary[word] = {}

                    self.word_summary[word][ref] = 1

    def count(self):
        return len(self.word_summary.keys())

    # DataFrame shape:
    # x-axis: coins
    # y-axis: reddit comment/article ids
    #
    def get_dataframe(self):
        df_merge = pd.DataFrame()
        for key in self.word_summary.keys():
            df = pd.DataFrame()
            df[key] = pd.Series([1]*len(self.word_summary[key].keys()), index=self.word_summary[key].keys())
            #print('word df {}'.format(df))

            df_merge = pd.concat([df_merge, df], axis=1)

        return df_merge

class CoinAssets(object):
    def __init__(self):
        # coin_assets.json retrieved from 
        self.df = pd.read_json('coin_assets.json')
        self.df = self.df.query('type_is_crypto == 1 & volume_1mth_usd > 0')
        self.df = self.df.sort_values(by=['volume_1mth_usd'],ascending=False).head(100)

    def get_list(self):
        assets = {}

        for asset in list(self.df['asset_id'].str.lower()):
            assets[str(asset).strip()] = 1

        for asset in list(self.df['name'].str.lower()):
            assets[str(asset).strip()] = 1

        return list(assets.keys())

if __name__ == '__main__':
    reddit = RedditApi()

    coins = CoinAssets()
    coins = coins.get_list()
    print('Found {} coin names'.format(len(coins)))
    #print(coins)

    me = reddit.about_me()
    print(me)

    #articles = reddit.get_articles('/r/wallstreetbets')
    articles = reddit.get_articles('/r/cryptomarkets')

    print('Fetched {} articles'.format(articles.count()))

    print('Writing reddit_articles.csv')
    with open('reddit_articles.csv', 'w+') as f:
        articles.df.to_csv(f)

    wf = WordFrequency(filter=coins)

    for name, row in articles.df[['title', 'selftext']].iterrows():
        words = nltk.word_tokenize(str(row['title']))
        wf.add_words(words, name)

        words = nltk.word_tokenize(str(row['selftext']))
        wf.add_words(words, name)

    comments_df = pd.DataFrame()

    for name in articles.df.index:
        comments = reddit.get_comments('r/cryptomarkets', name)

        print('Fetched {} comments'.format(comments.count()))

        for index, row in comments.df[['body']].iterrows():
            words = nltk.word_tokenize(str(row['body']))
            #print('{} {}'.format(row['name'], words))
            wf.add_words(words, index)

        comments_df = comments_df.append(comments.df, ignore_index=False)

    print('Writing reddit_comments.csv')
    with open('reddit_comments.csv', 'w+') as f:
        comments_df.to_csv(f)

    df = wf.get_dataframe()
    print('Found {} coin mentions'.format(len(df.columns)))

    sentiment_summary = {
        'id': [],
        'text': [],
        'coin': [],
        'sentiment': [],
    }

    for coin, mentions in df.items():
        print('{}\t{}'.format(coin, mentions.notna().sum()))

        for name, value in mentions[mentions.notna()].items():
            text_list = []

            try:
                comment_body = comments_df.loc[name]['body']
                text_list.append(comment_body)
                #print('comment: {} {}'.format(name, comment_body))
            except Exception as e:
                comment_body = None

            if not comment_body:
                # Try article text
                try:
                    article_body = articles.df.loc[name]
                    text_list.append(article_body['title'])
                    if article_body['selftext'] != '':
                        text_list.append(article_body['selftext'])
                        
                    #print('article: {} {}'.format(name, article_body['title']))
                except Exception as e:
                    print('{} text not found ({})'.format(name, e))

            for text in text_list:
                # Todo: Run finBERT on the text to predict sentiment

                sentiment_summary['id'].append(name)
                sentiment_summary['text'].append(text)
                sentiment_summary['coin'].append(coin)
                sentiment_summary['sentiment'].append(None)

    print('Writing word_frequency.csv')
    with open('word_frequency.csv', 'w+') as f:
        df.to_csv(f)

    print('Writing sentiment_summary.csv')
    with open('sentiment_summary.csv', 'w+') as f:
        df = pd.DataFrame.from_dict(sentiment_summary)
        df = df.set_index('id')
        df.to_csv(f)

