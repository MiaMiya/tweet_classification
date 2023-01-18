from fastapi import FastAPI
from google.cloud import storage
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import io
#my_model = pickle.loads(blob.download_as_string())

BUCKET_NAME = 'tweet_classification'
MODEL_FILE = 'my_trained_model.pt'

client = storage.Client()
bucket = client.get_bucket(BUCKET_NAME)
blob = bucket.get_blob(MODEL_FILE)
blob_io = io.BytesIO(blob.download_as_string())

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels = 2)
model.load_state_dict(torch.load(blob_io))
#model.load_state_dict(torch.load(/gcs/tweet_classification/raw))
#

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

app = FastAPI()

@app.get("/")
def read_root():
    return {"Intructions": "Go to /tweet/ and input a given tweet to see if it's Trump or Russian troll"}


@app.get("/tweet/{tweet}")
def read_item(tweet: str):
    tokenized_tweet = tokenizer(tweet, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
    outputs = model(
                input_ids=tokenized_tweet['input_ids'],
                attention_mask=tokenized_tweet['attention_mask'])
    log_probs = outputs.logits[0] ## input CR
    probs = log_probs.softmax(dim=-1).detach().cpu().flatten().numpy()
    if probs[0] > 0.5:
        pred = 'Russian'
        prob = str(round(probs[0],2))
    else:
        pred = 'Trump'
        prob = str(round(probs[1],2))

    response = {'Prediction': pred,
                  'With probability': prob,
                  'Tweet' : tweet}
    return response