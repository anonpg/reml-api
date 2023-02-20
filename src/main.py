from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

model = joblib.load('/app/data/20230220_baseline.xz')


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/predict")
def read_item(req: Request):
    params = dict(req.query_params)
    params = {
        'city2': np.nan,
        'city_plan': np.nan,
        'nearest_sta': np.nan,
        'nearest_sta_dist': np.nan,
        'floor_area': np.nan,
        'front_road_width': np.nan,
        **params,
    }
    df = pd.DataFrame([params])
    res = model.predict(df).iloc[0]
    return res


@app.get("/metadata")
def read_item():
    return convert_nan(model.metadata())


# https://qiita.com/Nabetani/items/1e9af1ee1d25e3b463a0
def convert_nan(x):
    if isinstance(x, dict):
        return { y: convert_nan(x[y]) for y in x }
    if isinstance(x, list):
        return [ convert_nan(y) for y in x ]

    if isinstance(x, float) and not np.isfinite(x):
        return None

    return x
