import sys
sys.path.append(r'./deps/finBERT')

import re
import pandas as pd

def do_predictions_celery():
    from reddit_tasks import predict
    max_predictions = 1000

    sentiments = pd.read_csv('sentiment_labels_clean.csv', low_memory=False)

    tasks = []
    predictions = []
    sentences = []
    scores = []

    # Testing first few sentences
    for idx, row in sentiments.head(max_predictions).iterrows():
        text = row['text']
        text = re.sub('^ATH| ATH', ' all time high', text, flags=re.I)
        text = re.sub('^hodl| hodl', ' hold', text, flags=re.I)
        text = re.sub('^DCA| dca', ' dollar cost average', text, flags=re.I)
        text = re.sub('^FTW| ftw', ' for the win', text, flags=re.I)
        text = re.sub('^ROI| roi', ' return on investment', text, flags=re.I)
        text = re.sub('^mooned| mooned', ' risen sharply in price', text, flags=re.I)
        text = re.sub('^to the moon| to the moon', ' price will rise sharply', text, flags=re.I)
        text = re.sub('^moonshot| moonshot', ' sharp price increase', text, flags=re.I)
        text = re.sub('^shitcoin| shitcoin', ' coin with no value', text, flags=re.I)

        # To-do - emoji conversion ![gif](emote|snoomoji_pack|shrug)
        tasks.append(predict.delay(text, idx))

    for task in tasks:
        results = task.get()
        #print(results)

        id = results['id']
        df = pd.DataFrame.from_dict(results['predictions'])
        #print(df)

        coin = sentiments.loc[id]['coin']
        #print(coin)

        # Text is split into sentences. Select the first sentence with a coin mention.
        prediction = df[df['sentence'].str.contains(coin, case=False)]
        prediction = prediction.iloc[0]
        #print(prediction)

        print('[{}/{}] {} {}'.format(id+1, len(sentiments), coin, prediction['prediction']))
        sentences.append(prediction['sentence'])
        predictions.append(prediction['prediction'])
        scores.append(prediction['sentiment_score'])

    if len(predictions) < max_predictions:
        for idx in range(max_predictions-len(predictions)):
            predictions.append(None)

    sentiments = sentiments.head(max_predictions)
    sentiments['sentence'] = sentences
    sentiments['prediction'] = predictions
    sentiments['score'] = scores

    sentiments.to_csv('sentiment_labels_predictions.csv')

def do_predictions():
    from finbert.finbert import predict
    from transformers import AutoModelForSequenceClassification
    model_path = 'deps/finBERT/models/classifier_model/finbert-sentiment'
    model = AutoModelForSequenceClassification.from_pretrained(model_path,num_labels=3,cache_dir=None)

    sentiments = pd.read_csv('sentiment_labels_clean.csv', low_memory=False)

    predictions = []
    # Testing first few sentences
    for idx, row in sentiments.head(1000).iterrows():
        results = predict(row['text'], model)
        # Text is split into sentences. Select the first sentence with a coin mention.
        results = results[results['sentence'].str.contains(row['coin'], case=False)]
        print(results['prediction'])
        prediction = list(results['prediction'])[0]
        #positive = results[results['prediction']=='positive']['prediction'].count()
        #negative = results[results['prediction']=='negative']['prediction'].count()
        predictions.append(prediction)

    #for idx in range(990):
        #predictions.append('neutral')

    sentiments['prediction'] = predictions

    sentiments.to_csv('sentiment_labels_predictions.csv')
    #results.to_csv('finbert_results.csv')
    #print('results {}'.format(results))

if __name__ == '__main__':
    #do_predictions()
    do_predictions_celery()