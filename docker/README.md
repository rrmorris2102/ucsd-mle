# Crypto Sentiment Analysis Deployment Guide

This guide outlines the steps for deploying the Crypto Sentiment Analysis application.  A test deployment is available at:

* Model API Server - http://025a16bba0cc.ngrok.io
* Web UI - http://20.102.58.252


## 1. Training
Run the xlnet_train.py script to generate the trained xlnet model to models/xlnet_model_batch48.bin.

Link to xlnet model trained with Reddit crypto comments:  
https://drive.google.com/file/d/1c8mTJkXmrr0UCmB4DWgi5ca7yl0QQAUP/view?usp=sharing

## 2. Source Tree
```
├── docker
│   ├── data_loader - Data Container
│   ├── sentiment_updater - Model Container
│   └── web_server - Web Container
├── docs
│   └── UCSD MLE Capstone Project.pdf
├── reddit - Source for Data Container
└── xlnet - Source for Model and Web Containers
```

## 3. Model API Server
Model API Server runs predictions on available GPUs and returns a sentiment summary. Use the following command to start the server:

```
cd xlnet
export AZURE_ACCOUNT_KEY=<your Azure account key>
python3 api.py
```

The following endpoints are available:
| URL | Method | Description |
|-----|--------|-------------|
| http://025a16bba0cc.ngrok.io/predict | POST | Run sentiment analysis on the provided text |
| http://025a16bba0cc.ngrok.io/summary | GET | Report positive, negative and neutral counts for crypto coins

Usage example:
```
import requests
import logging

class XLNetRequest(object):
    def __init__(self):
        #self.url = 'http://localhost:8000'
        self.url = 'http://025a16bba0cc.ngrok.io'

    def predict(self, body):
        data = {'body': body}
        x = requests.post(self.url + '/predict', json=data)
        return x.json()

    def summary(self):
        x = requests.get(self.url + '/summary')
        return x.json()

xlnet_api = XLNetRequest()

body = ['I love Bitcoin!', 'I hate Bitcoin!']

sentiment = xlnet_api.predict(body)
logging.info(sentiment)

summary = xlnet_api.summary()
logging.info(summary)        
```

##

## 4. Data Container
Data container scrapes comments from crypto Reddits and stores them in a Cosmos DB.  Steps to build and run:
* Add Azure key to docker/env
* Build the data-loader container
* Run data-loader container

Building the data-loader container:
```
cd docker/data-loader
sudo bash build.sh
```

Running the data-loader container:
```
cd docker/data-loader
sudo bash start.sh
```

## 5. Sentiment Container
Sentiment container adds sentiment to the comments in the Cosmos DB.  Steps to build and run:
* Build the sentiment-updater container
* Run sentiment-updater container

Building the sentiment-updater container:
```
cd docker/sentiment-updater
sudo bash build.sh
```

Running the sentiment-updater container:
```
cd docker/sentiment-updater
sudo bash start.sh
```
