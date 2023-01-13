FROM python:3.9-slim

# install python 
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

#### Files ####
COPY requirements.txt requirements.txt
COPY Makefile Makefile
COPY setup.py setup.py

##### folders ####
COPY src/ src/


WORKDIR /


ENTRYPOINT ["python", "-u", "src/models/train_model.py"]
