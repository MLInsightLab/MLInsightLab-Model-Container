FROM python:3.12

# Change the workdir
WORKDIR /code

# Copy and install python requirements
COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy all required scripts
COPY ./main.py /code/main.py
COPY ./run.sh /code/run.sh

# Expose port
EXPOSE 8888
