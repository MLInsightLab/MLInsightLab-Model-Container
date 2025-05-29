from transformers import pipeline, BitsAndBytesConfig
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import subprocess
import mlflow
import json
import os

# Model flavor variables
MODEL_FLAVOR = os.environ['MODEL_FLAVOR']
PYFUNC_FLAVOR = 'pyfunc'
SKLEARN_FLAVOR = 'sklearn'
TRANSFORMERS_FLAVOR = 'transformers'
HFHUB_FLAVOR = 'hfhub'

# Default location to save the model
model_location = './model'

# Base class for a prediction request


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
elif MODEL_FLAVOR == HFHUB_FLAVOR:

    # Get any keyword arguments or other parameters
    requirements = os.getenv('REQUIREMENTS')
    if requirements:
        subprocess.run(
            [
                'pip',
                'install',
                requirements
            ]
        )

    kwargs = os.getenv('KWARGS')
    if kwargs:
        kwargs = json.loads(kwargs)
    else:
        kwargs = {}

    quantization_kwargs = os.getenv('QUANTIZATION_KWARGS')
    if quantization_kwargs:
        quantization_kwargs = json.loads(quantization_kwargs)
        bnb_config = BitsAndBytesConfig(**quantization_kwargs)
        if not kwargs.get('model_kwargs'):
            kwargs['model_kwargs'] = {}
        kwargs['model_kwargs']['quantization_config'] = bnb_config

    model = pipeline(**kwargs)


# Set up the app
app = FastAPI(docs_url=None, redoc_url=None)

# Endpoint for prediction


@app.post('/predict')
def predict(body: PredictRequest):
    """
    Run prediction
    """

    # If convert to numpy and it's not a transformers model, convert accordingly
    if MODEL_FLAVOR not in [TRANSFORMERS_FLAVOR, HFHUB_FLAVOR] and body.convert_to_numpy:
        to_predict = np.array(body.data)
        if body.dtype:
            to_predict = to_predict.astype(body.dtype)
    else:
        to_predict = body.data

    # Set results as None as fallback
    results = None

    # If predict, run through the logic for running predict for the various flavors supported
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

        # If there's an exception thrown, try reshaping the array and running prediction again
        except Exception:
            try:
                if isinstance(to_predict, np.ndarray):
                    to_predict = to_predict.reshape(-1, 1)
                if MODEL_FLAVOR == PYFUNC_FLAVOR:
                    results = model.predict(to_predict, params=body.params)
                elif MODEL_FLAVOR in [TRANSFORMERS_FLAVOR, HFHUB_FLAVOR]:
                    if body.params:
                        results = model(to_predict, **body.params)
                    else:
                        results = model(to_predict)
                elif MODEL_FLAVOR == SKLEARN_FLAVOR:
                    results = model.predict(to_predict)

            # If the exception is not effectively handled, raise an exception stating what the issue is
            except Exception as e:
                raise ValueError(
                    f'There was an error running `predict`: {str(e)}'
                )

    # If the predict function is predict_proba, run the prediction logic
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

    # If the predict function is transform, run the prediction logic
    elif body.predict_function == 'transform':
        try:
            results = model.transform(to_predict)
        except Exception:
            try:
                results = model.transform(to_predict.reshape(-1, 1))
            except Exception as e:
                raise ValueError(
                    f'There was an error running `transform`: {str(e)}'
                )

    # If the predict function is not a supported one, raise exception and state the cause
    else:
        raise ValueError(
            'Only `predict` and `predict_proba` are supported prediction functions'
        )

    # Ensure response gets pushed back as a list for JSON compatibility
    if isinstance(results, np.ndarray):
        results = results.tolist()

    # Return response
    return {
        'prediction': results
    }
