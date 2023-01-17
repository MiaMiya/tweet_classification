from fastapi import FastAPI
from google.cloud import storage
import pickle
from src.models.model import get_model
from src.data.helper import tokenize_function
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
#my_model = pickle.loads(blob.download_as_string())

BUCKET_NAME = 'tweet_classification'
MODEL_FILE = 'my_trained_model.pt'

client = storage.Client()
bucket = client.get_bucket(BUCKET_NAME)
blob = bucket.get_blob(MODEL_FILE)
tokenized_dataset = tweet.map(tokenize_function, batched=True, remove_columns=['text'])

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels = 2)

model.load_state_dict(torch.load(blob.download_as_string()))

app = FastAPI()

@app.get("/")
def read_root():
    return {"Intructions": "Go to /tweet/ and input a given tweet to see if it's Trump or Russian troll"}


@app.get("/tweet/{tweet}")
def read_item(tweet: str):

    outputs = model(
                input_ids=tokenized_dataset['input_ids'],
                attention_mask=tokenized_dataset['attention_mask'])
    log_probs = outputs.logits[0] ## input CR
    probs = log_probs.softmax(dim=-1).detach().cpu().flatten().numpy()
    if probs <0.5:
        pred = 'Russian'
    else:
    #     pred = 'Trump'

    response = {'tweet' : tweet,
                'Prediction': 'tokenized_dataset'}
    return response