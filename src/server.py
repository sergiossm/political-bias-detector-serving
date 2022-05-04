import aiohttp
import asyncio
import uvicorn

from fastai.text import *
from sklearn.metrics import f1_score
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse


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


app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])


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


@app.route('/predict', methods=['POST'])
async def predict(request):
    data = await request.body()

    input_text = json.loads(data.decode('utf-8'))['input_text']
    pred = learn_c.predict(input_text)[2] * 100

    return JSONResponse({
        'Unidas Podemos': f'{pred[0]:.2f}%',
        'PSOE': f'{pred[3]:.2f}%',
        'Ciudadanos': f'{pred[1]:.2f}%',
        'PP': f'{pred[2]:.2f}%',
        'VOX': f'{pred[4]:.2f}%',
    })


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=8501, log_level="info")
