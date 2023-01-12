from tests import _PATH_DATA
from src.data.make_dataset import Tweets
import os.path
import pytest
import numpy as np 

@pytest.mark.skipif(not os.path.exists(f'{_PATH_DATA}/processed/train_processed.npy'), reason="Data files not found")
@pytest.mark.skipif(not os.path.exists(f'{_PATH_DATA}/processed/test_processed.npy'), reason="Data files not found")

def test_data():

    dataset_train = np.load(f'{_PATH_DATA}/processed/train_processed.npy', allow_pickle=True)
    dataset_test =np.load(f'{_PATH_DATA}/processed/test_processed.npy', allow_pickle=True)

    assert len(dataset_train[0]) == 185109, "Training dataset does not have the right size"
    assert len(dataset_test[0]) == 61704, "Testing dataset does not have the right size"
    assert len(dataset_train[0]) == len(dataset_train[1]), "Training dataset have uneven number of images and labels"
    assert len(dataset_test[0]) == len(dataset_test[1]), "Testing dataset have uneven number of images and labels"


if __name__ == "__main__":
    test_data()