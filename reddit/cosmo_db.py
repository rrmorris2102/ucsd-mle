import os
import sys
import json
import logging
import pandas as pd
from typing import Counter 
from azure.cosmos import CosmosClient, PartitionKey, exceptions

class CosmosConnection(object):
    def __init__(self, account_url=None, account_key=None):
        if account_url is None:
            #account_url = os.environ['ACCOUNT_URI']
            account_url = 'https://crypto-sentiment.documents.azure.com:443/'

        if account_key is None:
            account_key = os.getenv('AZURE_ACCOUNT_KEY', None)

            if account_key is None:
                raise Exception('AZURE_ACCOUNT_KEY environment variable not set')
          
            #account_key = '***REMOVED***'

        # Create a logger for the 'azure' SDK
        logger = logging.getLogger('azure')
        logger.setLevel(logging.WARNING)

        # Configure a console output
        handler = logging.StreamHandler(stream=sys.stdout)
        logger.addHandler(handler)

        self.client = CosmosClient(account_url, credential=account_key, logging_enable=True)

        self.containers = {
            'articles':        {'partition': '/subreddit_id', 'container': None},
            'comments':        {'partition': '/parent_id',    'container': None},
            'subreddits':      {'partition': '/subreddit_id', 'container': None},
            'authors':         {'partition': '/author_id',    'container': None},
            'coins':           {'partition': '/coin_id',      'container': None},
            'coins_articles':  {'partition': '/coin_id',      'container': None},
            'coins_comments':  {'partition': '/coin_id',      'container': None},
            #'comment_article': {'partition': '/article_id',   'container': None},
        }

class CosmosAssociation(object):
    def __init__(self, client, container):
        self.client = client
        self.container = container

    def set(self, container, source, target):
        name = '{}_{}'.format(self.container.get_name(), container.get_name())
        association = self.client.containers[name]['container']

        query = 'SELECT VALUE COUNT(1) FROM {} t1 where t1.{}_id = "{}" and t1.{}_id = "{}"'.format(
                        name,
                        self.container.get_name(), source,
                        container.get_name(), target)
    
        items = association.query_items(
            query=query,
            enable_cross_partition_query=True
        )

        item_count = 0
        num_updated = 0

        for item in items:
            item_count = item
        
        if item_count == 0:
            data = {
                '{}_id'.format(self.container.get_name()): source,
                '{}_id'.format(container.get_name()): target
            }

            association.upsert_item(data)

    def get(self, container, id=None):
        items = []

        name = '{}_{}'.format(self.container.get_name(), container.get_name())
        association = self.client.containers[name]['container']

        query = 'SELECT * FROM {}'.format(name)
        if not id is None:
            query = 'SELECT * FROM {} t1 where t1.{}_id = "{}"'.format(name, self.container.get_name(), id)

        for item in association.query_items(
                query=query,
                enable_cross_partition_query=True):
            items.append(item)
        return items

class CosmosContainer(object):
    def __init__(self, client, container, index, date_index = None):
        self.client = client
        self._name = container
        self.container = client.containers[container]['container']
        self.index = index
        self.date_index = date_index
        self.association = CosmosAssociation(client, self)

    def get_name(self):
        return self._name

    def count(self):
        item_count = 0
        items = self.container.query_items(
                query='SELECT VALUE COUNT(1) FROM c',
                enable_cross_partition_query=True)

        for item in items:
            item_count = item

        return item_count

    def get(self, id=None, db_id=None):
        items = []

        query = 'SELECT * FROM c'
        if not id is None:
            query = 'SELECT * FROM c t1 where t1.{} = "{}"'.format(self.index, id)
        if not db_id is None:
            query = 'SELECT * FROM c t1 where t1.id = "{}"'.format(db_id)

        for item in self.container.query_items(
                query=query,
                enable_cross_partition_query=True):
            #items.append(json.dumps(item, indent=True))
            items.append(item)
        return items

    def get_last(self, filter=None):
        if not self.date_index:
            return None

        items = []

        if not filter is None:
            query = 'SELECT * FROM c t1 where t1.{} ORDER BY t1.{} DESC OFFSET 0 LIMIT 1'.format(filter, self.date_index)
        else:
            query = 'SELECT * FROM c t1 ORDER BY t1.{} DESC OFFSET 0 LIMIT 1'.format(self.date_index)

        for item in self.container.query_items(
                query=query,
                enable_cross_partition_query=True):
            items.append(item)

        return items

    def get_dataframe(self):
        items = self.get()
        df = pd.DataFrame(items)
        index = 'id'
        if not self.index is None:
            index = self.index
        df.set_index(index, inplace=True, drop=True)
        return df

class RedditDBSubreddits(CosmosContainer):
    def __init__(self, client):
        super().__init__(client, 'subreddits', index='subreddit_id')

    # Add or update database record
    def add(self, data):
        items = self.container.query_items(
                    query='SELECT * FROM subreddits t1 where t1.subreddit_id = "{}"'.format(data['subreddit_id']),
                    enable_cross_partition_query=True, max_item_count=1)

        item_count = 0

        for item in items:
            item_count += 1
            logging.debug('item {}'.format(json.dumps(item, indent=True)))
            if item['subscribers'] != data['subscribers']:
                # Update the subscriber count
                item['subscribers'] = data['subscribers']
                self.container.replace_item(item=item, body=item)

        if item_count == 0:
            # New item
            self.container.upsert_item(data)
            item_count += 1

        return item_count

class RedditDBAuthors(CosmosContainer):
    def __init__(self, client):
        super().__init__(client, 'authors', index='author_id')

    # Add or update database record
    # Returns 0 if record already exists
    def add(self, data):
        items = self.container.query_items(
                    query='SELECT VALUE COUNT(1) FROM authors t1 where t1.author_id = "{}"'.format(data['author_id']))

        item_count = 0
        num_updated = 0

        for item in items:
            item_count = item
        
        if item_count == 0:
            self.container.upsert_item(data)
            num_updated = 1

        return num_updated

class RedditDBArticles(CosmosContainer):
    def __init__(self, client):
        super().__init__(client, 'articles', index='article_id', date_index='created_utc')

    # Add or update database record
    # Returns 0 if record already exists
    def add(self, data):
        items = self.container.query_items(
                    query='SELECT VALUE COUNT(1) FROM articles t1 where t1.article_id = "{}"'.format(data['article_id']),
                    enable_cross_partition_query=True)

        item_count = 0
        num_updated = 0

        for item in items:
            item_count = item
        
        if item_count == 0:
            self.container.upsert_item(data)
            num_updated = 1

        return num_updated

class RedditDBComments(CosmosContainer):
    def __init__(self, client):
        super().__init__(client, 'comments', index='comment_id', date_index='created_utc')

    # Add or update database record
    # Returns 0 if record already exists
    def add(self, data):
        items = self.container.query_items(
                    query='SELECT VALUE COUNT(1) FROM comments t1 where t1.comment_id = "{}"'.format(data['comment_id']),
                    enable_cross_partition_query=True)

        item_count = 0
        num_updated = 0

        for item in items:
            item_count = item
        
        if item_count == 0:
            self.container.upsert_item(data)
            num_updated = 1

        return num_updated

class RedditDBCoins(CosmosContainer):
    def __init__(self, client):
        super().__init__(client, 'coins', index='coin_id')

    # Add or update database record
    # Returns 0 if record already exists
    def add(self, data):
        items = self.container.query_items(
                    query='SELECT VALUE COUNT(1) FROM coins t1 where t1.coin_id = "{}"'.format(data['coin_id']),
                    enable_cross_partition_query=True)

        item_count = 0
        num_updated = 0

        for item in items:
            item_count = item
        
        if item_count == 0:
            self.container.upsert_item(data)
            num_updated = 1

        return num_updated

class RedditDB(object):
    def __init__(self, account_url=None, account_key=None):
        client = CosmosConnection(account_url, account_key)

        database_name = 'reddit'

        try:
            database = client.client.create_database(database_name)
        except exceptions.CosmosResourceExistsError:
            database = client.client.get_database_client(database_name)

        properties = database.read()
        logging.debug('database properties {}'.format(json.dumps(properties)))

        self.database = database
        self.client = client

        #self.__reset_db()

        for container_name, props in client.containers.items():
            try:
                container = database.create_container(id=container_name, partition_key=PartitionKey(path=props['partition']))
            except exceptions.CosmosResourceExistsError:
                container = database.get_container_client(container_name)
            except exceptions.CosmosHttpResponseError:
                raise

            props['container'] = container

        self.subreddits = RedditDBSubreddits(client)
        self.authors = RedditDBAuthors(client)
        self.articles = RedditDBArticles(client)
        self.comments = RedditDBComments(client)
        self.coins = RedditDBCoins(client)

        logging.debug(props)

    def __reset_db(self):
        logging.warning('__RESET_DB')
        for container in self.client.containers:
            self.database.delete_container(container)






