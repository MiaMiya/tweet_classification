# -*- coding: utf-8 -*-
import click
import logging
#from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from sklearn.model_selection import train_test_split
import click
import numpy as np
#import torch
from torch.utils.data import Dataset 

class Tweets(Dataset):
    def __init__(self, train: bool, in_folder: str = "", out_folder: str = "") -> None:
        super().__init__()
        
        self.in_folder = in_folder
        self.out_folder = out_folder
        self.train = train

        if self.out_folder: # try loading the preprocessed data 
            try: 
                self.load_preprocessed()
                print("Loaded the pre-processed files")
                return
            except ValueError: # Data not created yet and we will create it 
                pass

        # Loading the data 
        pd_Russian = pd.read_csv(in_folder + '/tweets.csv')
        pd_D_T = pd.read_csv(in_folder + '/realdonaldtrump.csv')

        pd_Russian['Label'] = 0
        pd_D_T['Label'] = 1

        pd_Russian.rename(columns = {'text' : 'Tweet'}, inplace=True)
        pd_D_T.rename(columns = {'content' : 'Tweet'}, inplace=True)

        # Combine the two dataframe
        pd_combine = pd.concat([pd_Russian[['Tweet','Label']],pd_D_T[['Tweet','Label']]], ignore_index=True).reset_index(drop=True)

        # Create train test split 
        Train, Test = train_test_split(pd_combine,
                                        random_state=104, 
                                        test_size=0.25, 
                                        shuffle=True,
                                        stratify=pd_combine['Label'])

        # Saving the train and validation data 
        self.train_tweet = Train['Tweet']
        self.train_label = Train['Label']
        self.test_tweet = Test['Tweet']
        self.test_label = Test['Label']

        if self.out_folder: 
            self.save_preprocessed()

    def load_preprocessed(self):
        try: 
            data_train = np.load(f"{self.out_folder}/train_processed.npy", allow_pickle=True)
            self.train_tweet = data_train[0,:]
            self.train_label = data_train[1,:]
            data_test = np.load(f"{self.out_folder}/test_processed.npy", allow_pickle=True)
            self.test_tweet = data_test[0,:]
            self.test_label = data_test[1,:]
        except:
            raise ValueError("No preprocessed files found")

    def save_preprocessed(self):
        np.save(self.out_folder + '/train_processed.npy', np.stack((self.train_tweet, self.train_tweet)), allow_pickle=True)
        np.save(self.out_folder + '/test_processed.npy', np.stack((self.test_tweet, self.test_label)), allow_pickle=True)

    # def __len__(self):
    #         return self.labels.numel()
        
    # def __getitem__(self, idx):
    #     return self.images[idx].float(), self.labels[idx]


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Load data
    data_set = Tweets(train=True, in_folder=input_filepath, out_folder=output_filepath)
    data_set.save_preprocessed()

    print(data_set.train_tweet.shape)
    print(data_set.train_label.shape)
    print(data_set.test_tweet.shape)
    print(data_set.test_label.shape)
    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    #project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
