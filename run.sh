# Check the flavor of the model - if hfhub, then don't download the model
if [ "$MODEL_FLAVOR" != "hfhub" ]; then

    # Download the model
    mlflow artifacts download --artifact-uri ${MODEL_URI} --dst-path ./model

    # Install python requirements for the model
    pip install --upgrade -r ./model/requirements.txt

fi

# Run the FastAPI app using uvicorn
uvicorn main:app --host 0.0.0.0 --port 8888
