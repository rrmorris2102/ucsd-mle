# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /app

COPY xlnet/requirements.txt requirements.txt
COPY xlnet/sentiment_updater.py sentiment_updater.py
COPY reddit/cosmo_db.py cosmo_db.py

RUN pip3 install -r requirements.txt

CMD [ "python3", "sentiment_updater.py"]
