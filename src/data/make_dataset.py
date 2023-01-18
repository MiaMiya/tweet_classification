# -*- coding: utf-8 -*-
import logging

import click
import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class Tweets(Dataset):
    def __init__(self, in_folder: str = "", out_folder: str = "") -> None:
        super().__init__()
        
        self.in_folder = in_folder
        self.out_folder = out_folder

        if self.out_folder: # try loading the preprocessed data 
            try: 
                self.load_preprocessed()
                print("Loaded the pre-processed files")
                return
            except ValueError: # Data not created yet and we will create it 
                print("No preprocessed data")
                pass

        # Loading the data 
        pd_Russian = pd.read_csv(in_folder + '/tweets.csv')
        pd_D_T = pd.read_csv(in_folder + '/realdonaldtrump.csv')

        pd_Russian['Label'] = 0
        pd_D_T['Label'] = 1

        pd_Russian.rename(columns = {'text' : 'Tweet'}, inplace=True)
        pd_D_T.rename(columns = {'content' : 'Tweet'}, inplace=True)
        
        # For real use
        # min_len_pd = min(len(pd_Russian), len(pd_D_T))

        # pd_Russian = pd_Russian.sample(frac = 1).iloc[:min_len_pd,:].reset_index(drop=True)
        # pd_D_T = pd_D_T.sample(frac = 1).iloc[:min_len_pd,:].reset_index(drop=True)

        # For making sure gcp works thus test with much smaller dataset
        pd_Russian = pd_Russian.sample(frac = 1).iloc[:500,:].reset_index(drop=True)
        pd_D_T = pd_D_T.sample(frac = 1).iloc[:500,:].reset_index(drop=True)

        # Combine the two dataframe
        pd_combine = pd.concat([pd_Russian[['Tweet','Label']],pd_D_T[['Tweet','Label']]], ignore_index=True).reset_index(drop=True)

        pd_combine.dropna(inplace=True)

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
            # For local 
            # data_train = np.load(f"{self.out_folder}/train_processed.npy", allow_pickle=True)
            # data_test = np.load(f"{self.out_folder}/test_processed.npy", allow_pickle=True)

            # ## Test 
            # BUCKET_NAME = "tweet_classification"
            # client = storage.Client()
            # bucket = client.get_bucket(BUCKET_NAME)
            # blob_train = bucket.get_blob('/data/processed/train_processed.npy')
            # blob_test = bucket.get_blob('/data/processed/test_processed.npy')
            # ## For gcp 
            # data_train = np.load(blob_train.download_as_string(), allow_pickle=True)
            # data_test = np.load(blob_test.download_as_string(), allow_pickle=True)

            data_train = np.load("/gcs/tweet_classification/processed/train_processed.npy", allow_pickle=True)
            data_test = np.load("/gcs/tweet_classification/processed/test_processed.npy", allow_pickle=True)

            # with open('gs://braided-destiny-374308/tweet_classification/data/processed/train_processed.npy', 'r') as f:
            #     data_train = f.readlines()
            # with open('gs://braided-destiny-374308/tweet_classification/data/processed/test_processed.npy', 'r') as f:
            #     data_test = f.readlines()

            self.train_tweet = data_train[0,:]
            self.train_label = data_train[1,:]
            self.test_tweet = data_test[0,:]
            self.test_label = data_test[1,:]
        except:
            raise ValueError("No preprocessed files found")

    def save_preprocessed(self):
        np.save(self.out_folder + '/train_processed.npy', np.stack((self.train_tweet, self.train_label)), allow_pickle=True)
        np.save(self.out_folder + '/test_processed.npy', np.stack((self.test_tweet, self.test_label)), allow_pickle=True)

    # def __len__(self):
    #         return self.labels.numel()
        
    # def __getitem__(self, idx):
    #     return self.images[idx].float(), self.labels[idx]


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    '''
    Runs data processing scripts to turn raw data from (../raw) into cleaned data ready to be analyzed (saved in ../processed).

            Parameters:
                    input_filepath (str): The filepath to the raw data (data/raw)
                    output_filepath (str): The filepath to where the preprocessed data should be saved (data/processed)

            Returns:
                    Nothing
    '''
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Load data
    data_set = Tweets(in_folder=input_filepath, out_folder=output_filepath)
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
