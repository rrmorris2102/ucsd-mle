import sys
from coin import CoinAssets
from crypto_sentiment import CryptoSentimenter, WordFrequency
from cosmo_db import RedditDB
import pandas as pd
from datetime import datetime, timezone
import multiprocessing, threading, queue

from reddit import RedditApi

import pprint
pp = pprint.PrettyPrinter(indent=4)

def test_reddit_web():
    reddit_db = RedditDB()

    print('{} coins'.format(reddit_db.coins.count()))
    print('{} subreddits'.format(reddit_db.subreddits.count()))
    print('{} authors'.format(reddit_db.authors.count()))
    print('{} articles'.format(reddit_db.articles.count()))
    print('{} comments'.format(reddit_db.comments.count()))

    print('')
    print('Coins:')
    coins_df = reddit_db.coins.get_dataframe()

    # Skip coin names that are common dictionary words
    skip_coins = ['ONE', 'MOON', 'LONG', 'BEAR', 'LINK', 'CASH', 'DOT', 'HOT', 'SUN', 'POT', 'BOX', 'LEND', 'DASH', 'MASK']
    coins_df = coins_df.drop(index=skip_coins)

    coins_df = coins_df.head(10)

    print('coin\tarticles\tcomments')
    count = []
    for coin in coins_df.index:
        coin_info = coins_df.loc[coin]

        articles = reddit_db.coins.association.get(reddit_db.articles, coin_info['id'])
        articles_count = 0
        if articles:
            articles_count = len(articles)

        comments = reddit_db.coins.association.get(reddit_db.comments, coin_info['id'])
        comments_count = 0
        if comments:
            comments_count = len(comments)

        count.append(articles_count + comments_count)

        if articles_count > 0 or comments_count > 0:
            print('{}\t{}\t{}'.format(coin, articles_count, comments_count))
    
    coins_df['count'] = count

    # Drop coins with no mentions
    coins_df = coins_df[(coins_df[['count']] != 0).all(axis=1)]

    coins_df = coins_df.sort_values(by=['count'], ascending=False)

    coins_df[['name', 'volume_1mth_usd', 'count']].to_csv('top_coins.csv')

    data = {
        'coin_id': [],
        'article_id': [],
        'title': [],
        'selftext': []
    }

    data_comments = {
        'coin_id': [],
        'comment_id': [],
        'body': []
    }

    for coin in coins_df.index:
        coin_info = coins_df.loc[coin]

        article_refs = reddit_db.coins.association.get(reddit_db.articles, coin_info['id'])
        if article_refs:
            for ref in article_refs[:10]:
                article = reddit_db.articles.get(db_id=ref['articles_id'])
                if article:
                    article = article[0]
                    data['coin_id'].append(coin)
                    data['article_id'].append(article['article_id'])
                    data['title'].append(article['title'])
                    data['selftext'].append(article['selftext'])

        comment_refs = reddit_db.coins.association.get(reddit_db.comments, coin_info['id'])
        if comment_refs:
            for ref in comment_refs[:10]:
                comment = reddit_db.comments.get(db_id=ref['comments_id'])
                if comment:
                    comment = comment[0]
                    data_comments['coin_id'].append(coin)
                    data_comments['comment_id'].append(comment['comment_id'])
                    data_comments['body'].append(comment['body'])

    articles_df = pd.DataFrame.from_dict(data)
    articles_df.to_csv('coins_articles.csv')

    comments_df = pd.DataFrame.from_dict(data_comments)
    comments_df.to_csv('coins_comments.csv')

def test_predict():
    from transformers import AutoModelForSequenceClassification
    sys.path.append(r'./deps/finBERT')

    from finbert.finbert import predict

    model_path = 'deps/finBERT/models/classifier_model/finbert-sentiment'
    model = AutoModelForSequenceClassification.from_pretrained(model_path,num_labels=3,cache_dir=None)

    sentimenter = CryptoSentimenter()
    sentimenter.scan('/r/cryptomarkets')
    #sentimenter.scan('/r/cryptocurrency')
    #sentimenter.scan('/r/cryptocurrencies')
    #sentimenter.scan('/r/cryptomoonshots')
    #sentimenter.scan('/r/satoshistreetbets')

    df = sentimenter.get_dataframe()

    # Testing first few sentences
    text = '\n'.join(list(df['text'])[:10])

    results = predict(text, model)
    results.to_csv('finbert_results.csv')
    #print('results {}'.format(results))

    print('Writing sentiment_summary.csv')
    with open('sentiment_summary.csv', 'w+') as f:
        df.to_csv(f)

def test_cosmo():
    #url = os.environ['ACCOUNT_URI']
    key = os.environ['AZURE_ACCOUNT_KEY']
    url = 'https://crypto-sentiment.documents.azure.com:443/'
    client = CosmosClient(url, credential=key)

    database_name = 'testDatabase'
    try:
        database = client.create_database(database_name)
    except exceptions.CosmosResourceExistsError:
        database = client.get_database_client(database_name)

    container_name = 'products'

    try:
        container = database.create_container(id=container_name, partition_key=PartitionKey(path="/productName"))
    except exceptions.CosmosResourceExistsError:
        container = database.get_container_client(container_name)
    except exceptions.CosmosHttpResponseError:
        raise

    for i in range(1, 10):
        container.upsert_item({
                'id': 'item{0}'.format(i),
                'productName': 'Widget',
                'productModel': 'Model {0}'.format(i)
            }
        )

    for item in container.query_items(
            query='SELECT * FROM products p WHERE p.id="item3"',
            enable_cross_partition_query=True):
        print(json.dumps(item, indent=True))

class MainProcess(multiprocessing.Process):
    def __init__(self):
        multiprocessing.Process.__init__(self)

    def run(self):
        #test_cosmo()
        #test_predict()
        test_reddit_web()

if __name__ == '__main__':
    main = MainProcess()
    main.start()
    main.join()    
