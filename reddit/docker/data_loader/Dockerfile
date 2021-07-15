# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
COPY cosmo_db.py cosmo_db.py
COPY test.py test.py
COPY data_loader.py data_loader.py
COPY coin.py coin.py
COPY reddit.py reddit.py
COPY crypto_sentiment.py crypto_sentiment.py
COPY coin_assets.json coin_assets.json

RUN pip3 install -r requirements.txt
RUN python3 -m nltk.downloader punkt

CMD [ "python3", "data_loader.py"]
