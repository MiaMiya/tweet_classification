import os.path

import pytest

from src.data.make_dataset import Tweets
from tests import _PATH_DATA


@pytest.mark.skipif(not os.path.exists(f'{_PATH_DATA}/processed/train_processed.npy'), reason="Data files not found")
@pytest.mark.skipif(not os.path.exists(f'{_PATH_DATA}/processed/test_processed.npy'), reason="Data files not found")
def test_data_len():
    dataset = Tweets(in_folder="data/raw",out_folder="data/processed")
    assert len(dataset.train_tweet) == len(dataset.train_label), "Training dataset have uneven number of images and labels"
    assert len(dataset.test_tweet) == len(dataset.test_label), "Testing dataset have uneven number of images and labels"

@pytest.mark.skipif(not os.path.exists(f'{_PATH_DATA}/processed/train_processed.npy'), reason="Data files not found")
@pytest.mark.skipif(not os.path.exists(f'{_PATH_DATA}/processed/test_processed.npy'), reason="Data files not found")
def test_data_type():
    dataset = Tweets(in_folder="data/raw",out_folder="data/processed")
    assert (isinstance(item, int) for item in dataset.train_label), "Non integer values found in train labels"
    assert (isinstance(item, int) for item in dataset.test_label), "Non integer values found in test labels"
    assert (isinstance(item, str) for item in dataset.train_tweet), "Non string values foudn for the training tweets"
    assert (isinstance(item, str) for item in dataset.test_tweet), "Non string values foudn for the testing tweets"
    assert set(dataset.train_label) == {1,0}, "Values other than 0 or 1 are found in train set labels"
    assert set(dataset.test_label) == {1,0}, "Values other than 0 or 1 are found in test set labels"

@pytest.mark.skipif(not os.path.exists(f'{_PATH_DATA}/processed/train_processed.npy'), reason="Data files not found")
@pytest.mark.skipif(not os.path.exists(f'{_PATH_DATA}/processed/test_processed.npy'), reason="Data files not found")
def test_data_na():
    dataset = Tweets(in_folder="data/raw",out_folder="data/processed")
    assert dataset.train_tweet.isnull().sum() == 0    

if __name__ == "__main__":
    test_data_len()
    test_data_type()
    test_data_na()