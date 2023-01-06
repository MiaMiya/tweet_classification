MLOps Project Description - Tweet Classification of Trump and Russian Troll Tweets
==============================

### Overall goal of the project
The goal of our project is to classify if a tweet was made by Trump or a Russian troll account. 

### What framework are you going to use (PyTorch Image Models, Transformer, Pytorch-Geometrics)
The input for our models is going to be text strings, thus Transformer framework will be used.

### How to you intend to include the framework into your project
The Transformer framework provide various pre-trained model and we will be selecting one of these to be the basis of our model. This will be included in our cookie cutter approach. 

### What data are you going to run on (initially, may change)
The data we have selected is tweets from two different Kaggle sources [Trump tweets](https://www.kaggle.com/datasets/austinreese/trump-tweets?resource=download) and [Russian troll tweets](https://www.kaggle.com/datasets/vikasg/russian-troll-tweets?select=tweets.csv). For each dataset we will include just the tweet and the date of the tweet. Initially we will only use the tweet (text) for the model. Then we will add a label to each tweet whether Trump or a Russian Troll is the author, based on which dataset the tweet comes from.   

### What deep learning models do you expect to use
We expect to used BERT model with focus on [BERT-base-uncased]([bert-base-uncased](https://huggingface.co/bert-base-uncased)), but if we have time, we will also checkout [BERTweet-base](https://huggingface.co/vinai/bertweet-base) which is trained on twitter data. 


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

## Checklist and report
==============================
See [reports/README.md](https://github.com/MiaMiya/tweet_classification/tree/main/reports)