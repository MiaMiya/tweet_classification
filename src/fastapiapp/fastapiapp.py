from fastapi import FastAPI
from google.cloud import storage
import pickle
from src.models.model import get_model
from src.data.helper import tokenize_function
import torch
#my_model = pickle.loads(blob.download_as_string())

BUCKET_NAME = 'tweet_classification'
MODEL_FILE = 'my_trained_model.pt'

client = storage.Client()
bucket = client.get_bucket(BUCKET_NAME)
blob = bucket.get_blob(MODEL_FILE)
model = get_model()
model.load_state_dict(torch.load(blob.download_as_string()))

app = FastAPI()


@app.get("/")
def read_root():
    return {"Intructions": "Go to /tweet/ and input a given tweet to see if it's Trump or Russian troll"}


@app.get("/tweet/{tweet}")
def read_item(tweet: str):
    tokenized_dataset = tweet.map(tokenize_function, batched=True, remove_columns=['text'])
    outputs = model(
                input_ids=tokenized_dataset['input_ids'],
                attention_mask=tokenized_dataset['attention_mask'])
    log_probs = outputs.logits[0] ## input CR
    probs = log_probs.softmax(dim=-1).detach().cpu().flatten().numpy()
    if probs <0.5:
        pred = 'Russian'
    else:
        pred = 'Trump'
    response = {'tweet' : tweet,
                'Prediction': pred}
    return response



# from enum import Enum
# class ItemEnum(Enum):
#     alexnet = "alexnet"
#     resnet = "resnet"
#     lenet = "lenet"

# @app.get("/restric_items/{item_id}")
# def read_item(item_id: ItemEnum):
#     return {"item_id": item_id}


# @app.get("/query_items")
# def read_item(item_id: int):
#     return {"item_id": item_id}


# database = {'username': [ ], 'password': [ ]}

# @app.post("/login/")
# def login(username: str, password: str):
#     username_db = database['username']
#     password_db = database['password']
#     if username not in username_db and password not in password_db:
#         with open('database.csv', "a") as file:
#             file.write(f"{username}, {password} \n")
#         username_db.append(username)
#         password_db.append(password)
#     return "login saved"


# from http import HTTPStatus
# import re


# @app.get("/text_model/")
# def contains_email(data: str):
#     regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
#     response = {
#         "input": data,
#         "message": HTTPStatus.OK.phrase,
#         "status-code": HTTPStatus.OK,
#         "is_email": re.fullmatch(regex, data) is not None
#     }
#     return response


# from enum import Enum
# from pydantic import BaseModel

# class DomainEnum(Enum):
#     gmail = "gmail"
#     hotmail = "hotmail"

# class Item(BaseModel):
#     email: str
#     domain: DomainEnum

# @app.post("/text_model/")
# def contains_email_domain(data: Item):
#     if data.domain is DomainEnum.gmail:
#         regex = r'\b[A-Za-z0-9._%+-]+@gmail+\.[A-Z|a-z]{2,}\b'
#     if data.domain is DomainEnum.hotmail:
#         regex = r'\b[A-Za-z0-9._%+-]+@hotmail+\.[A-Z|a-z]{2,}\b'
#     response = {
#         "input": data,
#         "message": HTTPStatus.OK.phrase,
#         "status-code": HTTPStatus.OK,
#         "is_email": re.fullmatch(regex, data.email) is not None
#     }
#     return response


# from fastapi import UploadFile, File
# from fastapi.responses import FileResponse
# import cv2
# from typing import Optional

# @app.post("/cv_model/")
# async def cv_model(data: UploadFile = File(...), h: Optional[int] = 28, w: Optional[int] = 28):
#     with open('image.jpg', 'wb') as image:
#         content = await data.read()
#         image.write(content)
#         image.close()

#     img = cv2.imread("image.jpg")
#     res = cv2.resize(img, (h, w))

#     cv2.imwrite('image_resize.jpg', res)

#     response = {
#         "input": data,
#         "output": FileResponse('image_resize.jpg'),
#         "message": HTTPStatus.OK.phrase,
#         "status-code": HTTPStatus.OK,
#     }
#     return response
