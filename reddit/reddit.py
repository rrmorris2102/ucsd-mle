import os
import requests
import requests.auth
from datetime import datetime
import pytz
import pandas as pd
import logging

class RedditArticles(object):
    def __init__(self, response):
        article_summary = {
            'subreddit': [],
            'subreddit_name_prefixed': [],
            'subreddit_subscribers': [],
            'subreddit_id': [],
            'title': [],
            'name': [],
            'author': [],
            'author_fullname': [],
            'created_utc': [],
            'selftext': [],
            'permalink': [],
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
            'parent_id': [],
            'author': [],
            'author_fullname': [],
            'created_utc': [],
            'permalink': [],
            'ups': [],
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
            logging.debug(response)
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
        logging.debug('url {}'.format(url))
        headers = {"User-Agent": self.user_agent}
        response = requests.get(url, headers=headers).json()

        if 'error' in response:
            raise Exception('Error {}: {}'.format(response['error'], response['message']))

        return RedditArticles(response)
    
    def get_comments(self, subreddit, article):
        if article.startswith('t3_'):
            article = article[3:]
        url = 'https://www.reddit.com/{}/comments/{}/.json'.format(subreddit, article)
        logging.debug('url {}'.format(url))
        headers = {"User-Agent": self.user_agent}

        try:
            response = requests.get(url, headers=headers).json()
        except Exception as e:
            logging.error('Error {} getting {}'.format(e, url))
            return None

        if 'error' in response:
            raise Exception('Error {}: {}'.format(response['error'], response['message']))

        return RedditComments(response)