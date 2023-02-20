FROM jupyter/datascience-notebook:python-3.10.6

USER root

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir \
    cloudpickle==2.0.0 \
    lightgbm==3.2.1 \
    fastparquet==2022.11.0 \
    catboost==1.1.1 \
    optuna==3.0.5 \
    scipy==1.8.1 \
    fastapi \
    uvicorn[standard]

ADD . /app
WORKDIR /app
CMD python -m src.main
