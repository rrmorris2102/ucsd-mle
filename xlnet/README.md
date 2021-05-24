# XLNet for Sentiment Classification

XLnet implementation adapted from https://medium.com/swlh/using-xlnet-for-sentiment-classification-cfa948e65e85 by Shanay Ghag.

Two classes are implemented:

- XLNetSentiment
- XLNetSentimentTrain

## XLNetSentiment

XLNetSentiment is used to classify text as positive, negative or neutral.

Methods:
- predict - Performance sentiment analysis

Usage example:

```python
def xlnet_predict():
    model_file = './models/xlnet_model.bin'
    xlnet = XLNetSentiment(model_file)

    text = "Movie is the worst one I have ever seen!! The story has no meaning at all"
    results = xlnet.predict(text)
    print(results)
```

## XLNetSentimentTrain

XLNetSentimentTrain is used to train the underlying XLNetForSequenceClassification model.

Methods:  
**train** - Perform training on the labeled dataframe.  The 'columns' argument specifies the names of the text and sentiment columns in the dataframe.  The sentiment column should encode the values as follows:

| Encoding | Description |
| --- | ----------- |
| 0 | positive |
| 1 | negative |
| 2 | neutral |


Usage example:

```python
def xlnet_train():
    df = get_data_set()
    xlnet_train = XLNetSentimentTrain()
    xlnet_train.train(df, columns=['review', 'sentiment'])
```

