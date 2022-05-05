from fastai.text import Path, warnings, load_learner
from flask import Flask, jsonify, request
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS

# classes = ['GCUP-EC-GC', 'GCs', 'GP', 'GS', 'GVOX']
#
path = Path(__file__).parent
models_dir = path / 'models'
models_dir.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "on'


app = Flask(__name__)
api = Api(app)
CORS(app)

# Define parser and request args
parser = reqparse.RequestParser()
parser.add_argument('input_text', type=str)


# Load model
learn_c = load_learner(models_dir)

# preds = torch.load(models_dir / 'preds.pt')
# y = torch.load(models_dir / 'y.pt')
# losses = torch.load(models_dir / 'losses.pt')
#
# ci = ClassificationInterpretation(learn_c, preds, y, losses)
# txt_ci = TextClassificationInterpretation(learn_c, preds, y, losses)


class status (Resource):
    def get(self):
        try:
            return {'data': 'Api is Running'}
        except:
            return {'data': 'An Error Occurred during fetching Api'}


class Predict(Resource):
    def post(self):
        json_data = request.get_json(force=True)
        input_text = json_data['input_text']
        pred = learn_c.predict(input_text)[2] * 100
        pred_list = pred.tolist()

        # asd = attention = txt_ci.intrinsic_attention(text=input_text)

        return jsonify(preds={
                'GCUP-EC-GC': pred_list[0],
                'GS': pred_list[3],
                'GCs': pred_list[1],
                'GP': pred_list[2],
                'GVOX': pred_list[4],
            })


api.add_resource(status, '/')
api.add_resource(Predict, '/predict')


if __name__ == '__main__':
    app.run()
