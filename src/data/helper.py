import torch

from src.models.model import get_tokenizer


def tokenize_function(examples):
    tokenizer = get_tokenizer()
    return tokenizer(
        examples["text"], max_length=512, padding="max_length", truncation=True
    )


def collate_fn(inputs):
    """
    Defines how to combine different samples in a batch
    """
    input_ids = torch.tensor([i["input_ids"] for i in inputs])
    attention_mask = torch.tensor([i["attention_mask"] for i in inputs])
    labels = torch.tensor([i["label"] for i in inputs])

    return {"input_ids": input_ids, "attention_mask": attention_mask, "label": labels}
