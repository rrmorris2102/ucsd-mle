from flask import Flask
from flask_restful import reqparse, Resource, Api
from xlnet import XLNetSentiment, XLNetSentimentTrain

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('body', action='append')

model_file = './models/xlnet_model_batch48.bin'
xlnet = XLNetSentiment(model_file, batchsize=1, max_len=64)

class Inference(Resource):
    def get(self):
        return {'hello': 'world'}

    def post(self):
        args = parser.parse_args()
        sentiment = []
        try:
            for body in args['body']:
                results = xlnet.predict(args['body'])
                sentiment.append(results['sentiment'])
            return {'sentiment': sentiment}, 201
        except Exception as e:
            return 500

api.add_resource(Inference, '/predict')

if __name__ == '__main__':
    app.run(debug=False, port=8000)