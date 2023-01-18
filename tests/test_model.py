import pytest
import torch

from src.models.model import get_model

# Try
def test_error_on_wrong_shape():
    model = get_model()
    # with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
    with pytest.raises((ValueError, RuntimeError)):
        model(torch.randn(1, 2, 3))


def test_model():
    model = get_model()
    assert model(
        input_ids=torch.arange(512).reshape(1, 512),
        attention_mask=torch.ones(1, 512, dtype=torch.int8),
    ).logits[0].shape == torch.Size([2]), "Model output logits dimention not [2]"


if __name__ == "__main__":
    test_model()
    test_error_on_wrong_shape()
