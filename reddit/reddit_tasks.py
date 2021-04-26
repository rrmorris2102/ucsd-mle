import sys
sys.path.append(r'./deps/finBERT')

import os
import time
from celery import Task, Celery
from finbert.finbert import predict as finbert_predict
from transformers import AutoModelForSequenceClassification

app = Celery('reddit_tasks', backend='rpc://', broker='pyamqp://guest@localhost//')

class ModelTask(Task):
    _model = None

    @property
    def model(self):
        if self._model is None:
            model_path = 'deps/finBERT/models/classifier_model/finbert-sentiment'
            model = AutoModelForSequenceClassification.from_pretrained(model_path,num_labels=3,cache_dir=None)
            self._model = model
        return self._model

@app.task(base=ModelTask, bind=True)
def predict(self, text, id):
    print('Executing task id {0.id}, args: {0.args!r} kwargs: {0.kwargs!r}'.format(
            self.request))

    prediction = finbert_predict(text, self.model)
    #print(prediction.info())

    predictions = []
    for idx, row in prediction.iterrows():
        predictions.append({'sentence': row['sentence'], 'prediction': row['prediction']})

    return {'id': id, 'predictions': predictions}