FROM python:3.9-slim

# install python 
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

#### Files ####
COPY requirements.txt requirements.txt
COPY Makefile Makefile
COPY setup.py setup.py
COPY data.dvc data.dvc

##### folders ####
COPY src/ src/


WORKDIR /

#### pip install ####
RUN pip install -r requirements.txt --no-cache-dir


ENTRYPOINT ["python", "-u", "src/models/train_model.py"]
