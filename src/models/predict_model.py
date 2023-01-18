# -*- coding: utf-8 -*-
import logging

import click
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from dotenv import find_dotenv, load_dotenv
from torch import nn
from torch.utils.data import DataLoader

from src.data.helper import collate_fn, tokenize_function
from src.models.model import get_model


def predict(
    model: nn.Module, 
    test_dl: DataLoader, 
    device: torch.device
):
    '''
    The main prediction loop which will optimize a given model on a given dataset

            Parameters:
                    model (nn.Module): The model being optimized
                    test_dl (DataLoader): The prediction dataset
                    device (torch.device): The device to train on

            Returns:
                    prediction (list): List of predicted labels
                    probability (list): List of probability for each prediction
    '''
    with torch.no_grad():
        model.eval()

        # Keep track of the prediction and probability
        prediction = []
        probability = []

        for tweet in test_dl:

            # Place each tensor on the GPU
            batch = {b: tweet[b].to(device) for b in tweet}

            # Pass the inputs through the model, get the current loss and logits
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'])

            log_probs = outputs.logits[0] ## input CR

            if device == torch.cuda.is_available():
                probs = log_probs.softmax(dim=-1).detach().cpu().flatten().numpy()
            else: 
                probs = log_probs.softmax(dim=-1).detach().flatten().numpy()
            
            probability.append(probs[0])
            prediction.append(probs[0] < 0.5)
            
    return prediction, probability

@click.command()
@click.option("--model_checkpoint", default="/gcs/tweet_classification/my_trained_model.pt")
@click.option("--data_to_predict", default="/gcs/tweet_classification/processed/test_processed.npy")
def predict_main(model_checkpoint, data_to_predict):
    '''
    The main prediction function, with the following tasks:
    Loading model (with trained parameters) and data
    Tokenizing data
    Calling the predict() function to do predictions
    Printing the prediction with probability of the input data_to_predict

            Parameters:
                    model_checkpoint (torch.save object): The saved model state_dict
                    data_to_predict (file-like object): The data to predict the author of

            Returns:
                    Nothing
    '''
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)
    
    # Load model 
    model = get_model()
    model.load_state_dict(torch.load(model_checkpoint))

    # Define device cpu or gpu
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    print(device)
    model.to(device)

    # Load data
    data_test = np.load(data_to_predict, allow_pickle=True)

    data_set = Dataset.from_pandas(pd.DataFrame({'text':data_test[0,:], 'label':data_test[1,:]}))

    # Process the data by tokenizing it
    tokenized_dataset = data_set.map(tokenize_function, remove_columns=['text'])

    trainloader = DataLoader(tokenized_dataset, collate_fn=collate_fn)

    # Train the model 
    prediction, probs = predict(
    model, 
    trainloader,
    device)

    print("Predictions")
    for index,pred in enumerate(prediction):
        print(
            f"tweet {index+1} predicted to be class {pred} with probability {probs[index]}"
        )

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    #project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    predict_main()
