docker run -dit --memory=1g --cpus="1.0" --net="host" --name reddit-sentiment-updater --env-file ../env rrmorris/reddit-sentiment-updater:v1.0.2
