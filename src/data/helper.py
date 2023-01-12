from src.models.model import get_tokenizer


def tokenize_function(examples):
    tokenizer = get_tokenizer()
    return tokenizer(examples["text"], max_length = 512, padding='max_length',truncation=True)
