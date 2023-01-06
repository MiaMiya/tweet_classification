# -*- coding: utf-8 -*-
import click
import logging
#from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from sklearn.model_selection import train_test_split
import click
import torch

@click.command()
@click.option("input_filepath", default = 'data/raw', help = 'Path of data', type=click.Path(exists=True))
@click.option("output_filepath", default = 'data/processed', help = 'Path of data', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Load data
    pd_Russian = pd.read_csv(input_filepath + '/tweets.csv')
    pd_D_T = pd.read_csv(input_filepath + '/realdonaldtrump.csv')
    print(pd_Russian)
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

    # Save the data for later acess 
    Train.to_csv(output_filepath + 'Train.csv')
    Test.to_csv(output_filepath + 'Test.csv')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    #project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
