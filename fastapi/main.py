import pickle
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import uvicorn
from pathlib import Path
import pandas as pd
from predict_results import PredictResults

app = FastAPI(title="TelecomChurn")

BASE_PATH = Path(__file__).resolve().parent
BEST_MODEL_PICKL_NAME = 'stack_clf'
PIPE_NAME = 'data_pipeline'
MODEL_PATH = str(BASE_PATH / BEST_MODEL_PICKL_NAME)
PIPE_PATH = str(BASE_PATH / PIPE_NAME)

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

@app.post("/predict")
async def predict(raw_json: dict)->dict:
    raw_data = pd.DataFrame.from_dict(raw_json)
    clean_data = pipe.transform(raw_data)
    churn_result = model.predict(clean_data)[0]
    churn_prob = model.predict_proba(clean_data)[0][1]
    
    churn_result_model = PredictResults(
        churn_result=churn_result,
        churn_probability=churn_prob
        )
    
    encoded_result = jsonable_encoder(churn_result_model)
    return JSONResponse(content=encoded_result)

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)