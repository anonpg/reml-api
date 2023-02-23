from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import pandas as pd
import numpy as np
import joblib
import dataset
import os


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


@app.get("/predict")
def read_predict(req: Request):
    params = dict(req.query_params)
    params = {
        'city2': np.nan,
        'city_plan': np.nan,
        'nearest_sta': np.nan,
        'nearest_sta_dist': np.nan,
        'floor_area': np.nan,
        'front_road_width': np.nan,
        'building_year': np.nan,
        **params,
    }
    df = pd.DataFrame([params])
    # TODO: test
    df.loc[df['type'] != '宅地(土地と建物)', 'floor_area'] = np.nan
    df.loc[~df['type'].isin(['宅地(土地と建物)', '中古マンション等']), 'building_year'] = np.nan
    res = model.predict(df).iloc[0]
    return res


@app.get("/metadata")
def read_metadata():
    return metadata


@app.get("/rakumachis")
def read_rakumachis():
    db = dataset.connect(os.getenv('DATABASE_URL'))
    return list(db['rakumachis'].find())


# https://qiita.com/Nabetani/items/1e9af1ee1d25e3b463a0
def convert_nan(x):
    if isinstance(x, dict):
        return { y: convert_nan(x[y]) for y in x }
    if isinstance(x, list):
        return [ convert_nan(y) for y in x ]

    if isinstance(x, float) and not np.isfinite(x):
        return None

    return x


model = joblib.load('/app/data/20230222_pos.xz')
metadata = convert_nan(joblib.load('/app/data/20230221_eda_metadata.xz'))
