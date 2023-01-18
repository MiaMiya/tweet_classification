FROM python:3.9-slim

EXPOSE $PORT


RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY Makefile Makefile
COPY setup.py setup.py
COPY app/fastapi.py app/fastapi.py


RUN pip install -r requirements.txt --no-cache-dir

CMD exec uvicorn app.fastapiapp:app --port $PORT --host 0.0.0.0 --workers 1