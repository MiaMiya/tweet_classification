
from tests import _PATH_DATA
from src.data.make_dataset import Tweets
from src.models.model import get_model
import torch
import pytest

# Try
def test_error_on_wrong_shape():
    model = get_model()
    #with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
    with pytest.raises((ValueError, RuntimeError)):
        model(torch.randn(1,2,3))


def test_model():
   model = get_model()
   
   assert model(input_ids=torch.radn(1,2,3), attention_mask=torch.radn(1,2,3),
                    labels=torch.radn(1)).shape == torch.Size([1,2]), "Model output dimention not [1, 2]"



if __name__ == "__main__":
    test_model()
    test_error_on_wrong_shape()

