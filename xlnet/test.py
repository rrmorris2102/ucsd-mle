from xlnet import XLNetSentiment, XLNetSentimentTrain

import pandas as pd
from sklearn.utils import shuffle
import re

def get_data_set():
    # Reading data using pandas
    path_to_data = 'archive.zip'
    print('Loading {}'.format(path_to_data))
    df = pd.read_csv(path_to_data)

    # Shuffle and Clip data
    df = shuffle(df)
    df = df[:24000]

    # Function to clean text. Remove tagged entities, hyperlinks, emojis
    def clean_text(text):
        text = re.sub(r"@[A-Za-z0-9]+", ' ', text)
        text = re.sub(r"https?://[A-Za-z0-9./]+", ' ', text)
        text = re.sub(r"[^a-zA-z.!?'0-9]", ' ', text)
        text = re.sub('\t', ' ',  text)
        text = re.sub(r" +", ' ', text)
        return text
    
    df['review'] = df['review'].apply(clean_text)

    # Function to convert labels to number.
    def sentiment2label(sentiment):
        if sentiment == 'positive':
            return 0
        elif sentiment == 'negative':
            return 1
        else: # neutral
            return 2

    df['sentiment'] = df['sentiment'].apply(sentiment2label)

    return df

def xlnet_train():
    df = get_data_set()
    xlnet_train = XLNetSentimentTrain()
    xlnet_train.train(df, ['review', 'sentiment'])

def xlnet_predict():
    model_file = './models/xlnet_model.bin'
    xlnet = XLNetSentiment(model_file, batchsize=1, max_len=64)

    text = "xrp is the biggest scam ever. if you have a profit on it, sell now and get out while you still can."
    results = xlnet.predict(text)
    print(results)

    text = '"Rich Dad Poor Dad" Author Predicts That Bitcoin Could Hit $1.2 Million.'
    results = xlnet.predict(text)
    print(results)

#xlnet_train()
xlnet_predict()

