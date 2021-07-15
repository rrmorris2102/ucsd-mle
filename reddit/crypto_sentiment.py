import string
import pandas as pd
import nltk
from coin import CoinAssets
from reddit import RedditArticles, RedditComments, RedditApi

class WordFrequency(object):
    def __init__(self, filter=None):
        self.stop_words = ['b', 'i', 'are', 'as', 'but', 'we', 'an', 'by', 'has', 'or', 'was', 'that', 'me', 'like', 'have', 'if', 'so', 'your', 'the', 'and', 'for', 'on', 'to', 'I', 'a', 'of', 'with', 'is', 'my', 'you', 'be', 'in', 'this', 'at', 'it', 'from', 'after', 'their', '-']
        self.word_summary = {}
        self.df = None
        self.table = str.maketrans(dict.fromkeys(string.punctuation))
        self.filter = filter
    
    def add_words(self, words, ref):
        # https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string        

        words = nltk.word_tokenize(str(words))

        for word in words:
            word = word.strip().lower().translate(self.table)

            if word == '':
                continue

            if not word in self.stop_words:
                if not self.filter or (self.filter and word in self.filter):
                    if not word in self.word_summary:
                        self.word_summary[word] = {}

                    self.word_summary[word][ref] = 1

    def clear(self):
        self.word_summary.clear()

    def count(self):
        return len(self.word_summary.keys())

    @property
    def word_list(self):
        return list(self.word_summary.keys())

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

class CryptoSentimenter(object):
    def __init__(self):
        self.sentiment_summary = {
            'id': [],
            'text': [],
            'coin': [],
            'sentiment': [],
        }

        self.reddit = RedditApi()

        coins = CoinAssets()
        coins = coins.get_list()
        print('Found {} coin names'.format(len(coins)))
        self.coins = coins

    def scan(self, subreddit):
        articles = self.reddit.get_articles(subreddit)

        print('Fetched {} articles'.format(articles.count()))

        file_prefix = subreddit
        if file_prefix.startswith('/r/'):
            file_prefix = file_prefix[3:]

        fname = '{}_articles.csv'.format(file_prefix)
        print('Writing {}'.format(fname))
        with open(fname, 'w+') as f:
            articles.df.to_csv(f)

        wf = WordFrequency(filter=self.coins)

        for name, row in articles.df[['title', 'selftext']].iterrows():
            #words = nltk.word_tokenize(str(row['title']))
            wf.add_words(row['title'], name)

            #words = nltk.word_tokenize(str(row['selftext']))
            wf.add_words(row['selftext'], name)

        comments_df = pd.DataFrame()

        for name in articles.df.index:
            comments = self.reddit.get_comments(subreddit, name)

            if comments:
                print('Fetched {} comments'.format(comments.count()))

                for index, row in comments.df[['body']].iterrows():
                    #words = nltk.word_tokenize(str(row['body']))
                    #print('{} {}'.format(row['name'], words))
                    wf.add_words(row['body'], index)

                comments_df = comments_df.append(comments.df, ignore_index=False)

        fname = '{}_comments.csv'.format(file_prefix)
        print('Writing {}'.format(fname))
        with open(fname, 'w+') as f:
            comments_df.to_csv(f)

        df = wf.get_dataframe()
        print('Found {} coin mentions'.format(len(df.columns)))

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

                if comment_body is None:
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

                    self.sentiment_summary['id'].append(name)
                    self.sentiment_summary['text'].append(text)
                    self.sentiment_summary['coin'].append(coin)
                    self.sentiment_summary['sentiment'].append(None)

        fname = '{}_word_frequency.csv'.format(file_prefix)
        print('Writing {}'.format(fname))
        with open(fname, 'w+') as f:
            df.to_csv(f)

    def get_dataframe(self):
        df = pd.DataFrame.from_dict(self.sentiment_summary)
        df = df.set_index('id')
        return df