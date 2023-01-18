FROM python:3.9-slim

EXPOSE $PORT

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt --no-cache-dir

COPY fastapiapp.py fastapiapp.py

CMD exec uvicorn fastapiapp:app --port $PORT --host 0.0.0.0 --workers 1