import requests
import requests.auth
import os
import string
from datetime import datetime
import pytz
import pandas as pd

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
                    article_summary[column].append(post['data'][column].encode("utf-8"))
                elif column == 'created_utc':
                    dt_utc = datetime.fromtimestamp(post['data'][column])
                    dt_utc = dt_utc.astimezone(pytz.utc)
                    article_summary[column].append(dt_utc)
                else:
                    article_summary[column].append(post['data'][column])

        df = pd.DataFrame.from_dict(article_summary)
        #for key in article_summary.keys():
            #df[key] = pd.Series(article_summary[key])

        self.df = df
    
class RedditComments(object):
    def __init__(self, response):
        comment_summary = {
            'body': [],
            'author_fullname': [],
            'created_utc': []
        }

        def collect_comments(comment):
            for column in comment_summary.keys():
                #print('comment {}'.format(comment['data'].keys()))
                if column in comment['data']:
                    if column == 'created_utc':
                        dt_utc = datetime.fromtimestamp(comment['data'][column])
                        dt_utc = dt_utc.astimezone(pytz.utc)
                        comment_summary[column].append(dt_utc)
                    else:
                        comment_summary[column].append(comment['data'][column].encode("utf-8"))

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

        self.df = df

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
    def __init__(self):
        self.stop_words = ['i', 'are', 'as', 'but', 'we', 'an', 'by', 'has', 'or', 'was', 'that', 'me', 'like', 'have', 'if', 'so', 'your', 'the', 'and', 'for', 'on', 'to', 'I', 'a', 'of', 'with', 'is', 'my', 'you', 'be', 'in', 'this', 'at', 'it', 'from', 'after', 'their', '-']
        self.word_summary = {}
        self.df = None
        self.table = str.maketrans(dict.fromkeys(string.punctuation))
    
    def add_words(self, words):
        # https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string        

        for word in words:
            word = word.strip().lower().translate(self.table)

            if word == '':
                continue

            if not word in self.stop_words:
                if not word in self.word_summary:
                    self.word_summary[word] = 1
                else:
                    self.word_summary[word] += 1

    def count(self):
        return len(self.word_summary.keys())

    def get_dataframe(self):
        df = pd.DataFrame.from_dict(self.word_summary, orient='index', columns=['count'])
        df = df.sort_values(by=['count'], ascending=False)
        return df
        
if __name__ == '__main__':
    reddit = RedditApi()

    me = reddit.about_me()
    print(me)

    articles = reddit.get_articles('/r/wallstreetbets')

    with open('reddit_articles.csv', 'w+') as f:
        articles.df.to_csv(f)

    wf = WordFrequency()

    for title in articles.df['title']:
        words = str(title).split(' ')
        wf.add_words(words)

    for selftext in articles.df['selftext']:
        words = str(selftext).split(' ')
        wf.add_words(words)

    comments_df = pd.DataFrame()

    for name in articles.df['name']:
        comments = reddit.get_comments('r/wallstreetbets', name)

        for body in comments.df['body']:
            words = str(body).split(' ')
            wf.add_words(words)

        comments_df = comments_df.append(comments.df, ignore_index=True)

    with open('reddit_comments.csv', 'w+') as f:
        comments_df.to_csv(f)

    print('Found {} unique words'.format(wf.count()))

    df = wf.get_dataframe().head(50)
    for index, row in df.iterrows():
        print('{}\t{}'.format(index, row['count']))
    
    with open('word_frequency.csv', 'w+') as f:
        df.to_csv(f)

