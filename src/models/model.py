from transformers import AutoModelForSequenceClassification, AutoTokenizer


def get_model():
    return AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels = 2)#.to(device)

def get_tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-uncased")

