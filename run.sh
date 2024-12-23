# Download the model
mlflow artifacts download --artifact-uri ${MODEL_URI} --dst-path ./model

# Install python requirements for the model
pip install --upgrade -r ./model/requirements.txt

# Run the uvicorn app
uvicorn main:app --host 0.0.0.0 --port 8888
