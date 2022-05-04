import aiohttp
import asyncio

from fastai.text import *
from sklearn.metrics import f1_score
from flask import Flask, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS

export_file_url = 'https://www.dropbox.com/s/za657ddlzrvddth/export.pkl?raw=1'
export_file_name = 'export.pkl'

classes = ['GCUP-EC-GC', 'GCs', 'GP', 'GS', 'GVOX']

path = Path(__file__).parent
models_dir = path / 'models'
models_dir.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "on'


@np_func
def f1(inp, targ):
    return f1_score(targ, np.argmax(inp, axis=-1), average='weighted')


app = Flask(__name__)
api = Api(app)
CORS(app)


async def download_file(url, dest):
    if dest.exists():
        return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, models_dir / export_file_name)

    try:
        learn = load_learner(models_dir)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn_c = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


class Status(Resource):
    def get(self):
        try:
            return {'data': 'Api is Running'}
        except:
            return {'data': 'An Error Occurred during fetching Api'}


api.add_resource(Status, '/')

if __name__ == '__main__':
    app.run()

#
#
# @app.get("/")
# def read_root():
#     return {"Hello": "World"}
#
#
# @app.post("/items/")
# async def create_item(item: Item):
#     pred = learn_c.predict(item.text)[2] * 100
#
#     return {
#         'Unidas Podemos': f'{pred[0]:.2f}%',
#         'PSOE': f'{pred[3]:.2f}%',
#         'Ciudadanos': f'{pred[1]:.2f}%',
#         'PP': f'{pred[2]:.2f}%',
#         'VOX': f'{pred[4]:.2f}%',
#     }
