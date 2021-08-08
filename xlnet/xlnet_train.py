import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from xlnet import XLNetSentimentTrain

# Load labeled crypto sentiment data
path_to_data = '../reddit/sentiment_labels_clean.csv'
df = pd.read_csv(path_to_data)

# clip data
df = df[:1000]

# Drop columns that are not used for training
df = df.drop(['id', 'coin'], axis=1)

# Function to convert labels to number.
def sentiment2label(sentiment):
    if sentiment == 'positive':
        return 0
    elif sentiment == 'negative':
        return 1
    else: # neutral
        return 2

df['sentiment'] = df['sentiment'].apply(sentiment2label)

print(df.info())
print(df.head(10))

# Run training on the labeled crypto data - find the best sequence len
#sequences=[256, 128, 64]
sequences=[64]
history = []
for seq in sequences:
    xlnet_train = XLNetSentimentTrain(batchsize=48, max_len=seq)
    history.append(xlnet_train.train(df, ['text', 'sentiment']))

for hist in history:
    print('Epochs: {}'.format(hist['epochs']))
    print('Batchsize: {}'.format(hist['batchsize']))
    print('Max len: {}'.format(hist['max_len']))
    print(hist['classification_report'])
    print()
