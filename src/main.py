from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import pandas as pd
import numpy as np
import joblib
import dataset
import os
from functools import cache


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
app.add_middleware(GZipMiddleware)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
async def read_predict(req: Request):
    params_list = await req.json()
    params_list = list(map(_process_predict_params, params_list))

    # params = process_predict_params(dict(req.query_params))

    df = pd.DataFrame(params_list)
    # TODO: test
    df.loc[df['type'] != '宅地(土地と建物)', 'floor_area'] = np.nan
    df.loc[~df['type'].isin(['宅地(土地と建物)', '中古マンション等']), 'building_year'] = np.nan
    # res = model.predict(df).iloc[0]
    res = _get_model().predict(df).to_dict('records')
    return res


@app.get("/metadata")
def read_metadata():
    return _get_metadata()


@app.get("/rakumachis")
def read_rakumachis():
    return _get_rakumachis()


# https://qiita.com/Nabetani/items/1e9af1ee1d25e3b463a0
def convert_nan(x):
    if isinstance(x, dict):
        return { y: convert_nan(x[y]) for y in x }
    if isinstance(x, list):
        return [ convert_nan(y) for y in x ]

    if isinstance(x, float) and not np.isfinite(x):
        return None

    return x


def _process_predict_params(x):
    return {
        'city2': np.nan,
        'city_plan': np.nan,
        'nearest_sta': np.nan,
        'nearest_sta_dist': np.nan,
        'floor_area': np.nan,
        'front_road_width': np.nan,
        'building_year': np.nan,
        **x,
    }


@cache
def _get_model():
    return joblib.load('/app/data/20230222_pos.xz')


@cache
def _get_metadata():
    return convert_nan(joblib.load('/app/data/20230222_pos_metadata.xz'))


@cache
def _get_rakumachis():
    db = dataset.connect(os.getenv('DATABASE_URL'))
    return list(db['rakumachis'].find())
