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
We expect to used BERT model with focus on [BERT-base-uncased](https://huggingface.co/bert-base-uncased), if time allows, we will also checkout [BERTweet-base](https://huggingface.co/vinai/bertweet-base) which is trained on twitter data. 


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   ├── README.md      <- Exam questions with checklist for the project
    │   │
    │   ├── report.html    <- html version of the report
    │   │
    │   ├── report.py      <- Python file for testing contrains on the report format
    │   │
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── requirements_tests.txt   <- The requirements file for reproducing the tests environment, e.g.
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── make_dataset.py
    │   │   └── helper.py 
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── model.py
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │
    ├── .dvc               <- DVC for the project
    │   └── config         <- Configuration for dvc 
    │  
    ├── tests              <- Tests for code intergration 
    │   ├── __init__.py    <- Makes tests a Python module
    │   │
    │   ├── test_data.yaml <- Fort testing the training data
    │   │
    │   └── test_model.py  <- For testing the model 
    │
    ├── .github            <- For creation of CI in github
    │   ├── workflows           <- Scripts to download or generate data
    │   │   ├── caching.yaml    <- Test in different operation system and versions 
    │   │   ├── isort.yaml      <- Sorting and removing unused imports 
    │   │   └── flake8.yaml     <- Create test to compile iwth pep8
    │
    ├── app                <- Fastapi for deployment 
    │   ├── __init__.py    <- Makes app a Python module
    │   │
    │   └── fastapiapp.py  <- python code for createing the app using fastapi
    │
    ├── app.dockerfile     <- Docker file for fastapi
    ├── test.dockerfile    <- Docker file for inference
    ├── train.dockerfile   <- Docker file for training
    ├── couldbuild.yaml    <- Command for building docker images in cloud
    ├── config_cpu_fast.yaml    <- Configuring the run in cloud using cpu for fast api
    ├── config_cpu_inference.yaml   <- Configuring the run in cloud using cpu for inference
    ├── config_cpu_trian.yaml   <- Configuring the run in cloud using cpu for training model
    ├── data.dvc           <- Information on data in dvc
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

## Checklist and report
See [reports/README.md](https://github.com/MiaMiya/tweet_classification/tree/main/reports)
