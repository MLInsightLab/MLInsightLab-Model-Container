FROM python:3.12

# Change the workdir
WORKDIR /code

# Copy and install basic python requirements
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade pip uv && \
    uv pip install --system --no-cache-dir --upgrade -r /code/requirements.txt

# Copy all huggingface model requirements
COPY ./hfhub_requirements.txt /code/hfhub_requirements.txt
RUN uv pip install --system --upgrade -r /code/hfhub_requirements.txt

# Copy all required scripts
COPY ./main.py /code/main.py
COPY ./run.sh /code/run.sh

# Expose port
EXPOSE 8888

# Run the service
CMD [ "/bin/sh", "/code/run.sh" ]
