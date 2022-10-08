import pickle
from fastapi import FastAPI, Request
from pydantic import Json
import uvicorn
from fastapi.templating import Jinja2Templates
from pathlib import Path
import pandas as pd

app = FastAPI(title="TelecomChurn")

BASE_PATH = Path(__file__).resolve().parent
BEST_MODEL_PICKL_NAME = 'stack_clf'
PIPE_NAME = 'data_pipeline'
MODEL_PATH = BASE_PATH + 'Pickles/' + BEST_MODEL_PICKL_NAME
PIPE_PATH = BASE_PATH + 'Pickles/' + PIPE_NAME

TEMPLATES = Jinja2Templates(directory=str(BASE_PATH / "dashboard"))

try:
    with open(MODEL_PATH, 'rb') as handle:
        model = pickle.load(handle)
except EnvironmentError:
    print('cannot find model')

try:
    with open(PIPE_PATH, 'rb') as handle:
        pipe = pickle.load(handle)
except EnvironmentError:
    print('cannot find pipe')

@app.get("/")
async def root(request: Request) -> dict:
    result = 0
    return TEMPLATES.TemplateResponse(
        "index.html",
        {"request": request,'result': result},
    )

@app.post("/")
async def predict(raw_json:Json):
    raw_data = json.loads(raw_json)
    raw_data = pd.read_json(raw_data)
    clean_data = pipe.transform(raw_data)
    probability = model.predict_proba(clean_data)
    churn = model.predict(clean_data)

@app.get("/eda")
async def root(request: Request) -> dict:
    return TEMPLATES.TemplateResponse(
        "EDA.html",
        {"request": request},
    )
    
@app.get("/model")
async def root(request: Request) -> dict:
    return TEMPLATES.TemplateResponse(
        "model.html",
        {"request": request},
    )

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)