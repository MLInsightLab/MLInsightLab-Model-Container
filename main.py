from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import mlflow
import json
import os

MODEL_FLAVOR = os.environ['MODEL_FLAVOR']

PYFUNC_FLAVOR = 'pyfunc'
SKLEARN_FLAVOR = 'sklearn'
TRANSFORMERS_FLAVOR = 'transformers'

model_location = './model'


class PredictRequest(BaseModel):
    data: list
    predict_function: str = 'predict'
    convert_to_numpy: bool = True
    params: dict | None = None
    dtype: str | None = None


# Load the model
if MODEL_FLAVOR == PYFUNC_FLAVOR:
    model = mlflow.pyfunc.load_model(model_location)
elif MODEL_FLAVOR == SKLEARN_FLAVOR:
    model = mlflow.sklearn.load_model(model_location)
elif MODEL_FLAVOR == TRANSFORMERS_FLAVOR:
    if mlflow.transformers.is_gpu_available():
        model = mlflow.transformers.load_model(
            model_location,
            kwargs={
                'device_map': 'auto'
            }
        )
    else:
        model = mlflow.transformers.load_model(
            model_location,
        )

app = FastAPI(docs_url=None, redoc_url=None)


@app.post('/predict')
def predict(body: PredictRequest):

    if MODEL_FLAVOR != TRANSFORMERS_FLAVOR and body.convert_to_numpy:
        to_predict = np.array(body.data)
        if body.dtype:
            to_predict = to_predict.astype(body.dtype)
    else:
        to_predict = body.data

    results = None

    if body.predict_function == 'predict':
        try:
            if MODEL_FLAVOR == PYFUNC_FLAVOR:
                results = model.predict(to_predict, params=body.params)
            elif MODEL_FLAVOR == TRANSFORMERS_FLAVOR:
                if body.params:
                    results = model(to_predict, **body.params)
                else:
                    results = model(to_predict)
            elif MODEL_FLAVOR == SKLEARN_FLAVOR:
                results = model.predict(to_predict)
        except Exception:
            try:
                if isinstance(to_predict, np.ndarray):
                    to_predict = to_predict.reshape(-1, 1)
                if MODEL_FLAVOR == PYFUNC_FLAVOR:
                    results = model.predict(to_predict, params=body.params)
                elif MODEL_FLAVOR == TRANSFORMERS_FLAVOR:
                    if body.params:
                        results = model(to_predict, **body.params)
                    else:
                        results = model(to_predict)
                elif MODEL_FLAVOR == SKLEARN_FLAVOR:
                    results = model.predict(to_predict)
            except Exception as e:
                raise ValueError(
                    f'There was an error running `predict`: {str(e)}'
                )

    elif body.predict_function == 'predict_proba':
        try:
            results = model.predict_proba(to_predict)
        except Exception:
            try:
                results = model.predict_proba(to_predict.reshape(-1, 1))
            except Exception as e:
                raise ValueError(
                    f'There was an error running `predict_proba`: {str(e)}'
                )

    else:
        raise ValueError(
            'Only `predict` and `predict_proba` are supported prediction functions'
        )

    if isinstance(results, np.ndarray):
        results = results.tolist()

    return {
        'prediction': results
    }
