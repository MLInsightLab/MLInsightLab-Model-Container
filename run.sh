# Check the flavor of the model - if hfhub, then don't download the model
if [ "$MODEL_FLAVOR" != "hfhub" ]; then
    # Download the model
    printf "INFO $(date +'%Y-%m-%d %H:%M:%S'): DOWNLOADING MODEL\n"
    mlflow artifacts download --artifact-uri ${MODEL_URI} --dst-path ./model

    # Install python requirements for the model
    printf "INFO $(date +'%Y-%m-%d %H:%M:%S'): INSTALLING MODEL DEPENDENCIES\n"
    uv pip install --system --upgrade -r ./model/requirements.txt

else
    # Install the additional requirements
    printf "INFO $(date +'%Y-%m-%d %H:%M:%S'): INSTALLING MODEL DEPENDENCIES\n"
    uv pip install --system --upgrade -r /code/hfhub_requirements.txt

fi

# Run the FastAPI app using uvicorn
printf "INFO $(date +'%Y-%m-%d %H:%M:%S'): STARTING SERVER PROCESS\n"
uvicorn main:app --host 0.0.0.0 --port 8888
