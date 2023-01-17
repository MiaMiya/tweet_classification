---
layout: default
nav_exclude: true
---

# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

where you instead should add your answers. Any other changes may have unwanted consequences when your report is auto
generated in the end of the course. For questions where you are asked to include images, start by adding the image to
the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

will generate an `.html` page of your report. After deadline for answering this template, we will autoscrape
everything in this `reports` folder and then use this utility to generate an `.html` page that will be your serve
as your final handin.

Running

```bash
python report.py check
```

will check your answers in this template against the constrains listed for each question e.g. is your answer too
short, too long, have you included an image when asked to.

For both functions to work it is important that you do not rename anything. The script have two dependencies that can
be installed with `pip install click markdown`.

## Overall project checklist

The checklist is *exhaustic* which means that it includes everything that you could possible do on the project in
relation the curricilum in this course. Therefore, we do not expect at all that you have checked of all boxes at the
end of the project.

### Week 1

* [x] Create a git repository
* [x] Make sure that all team members have write access to the github repository
* [x] Create a dedicated environment for you project to keep track of your packages
* [x] Create the initial file structure using cookiecutter
* [x] Fill out the `make_dataset.py` file such that it downloads whatever data you need and
* [x] Add a model file and a training script and get that running
* [x] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
* [ ] Remember to comply with good coding practices (`pep8`) while doing the project
* [ ] Do a bit of code typing and remember to document essential parts of your code
* [x] Setup version control for your data or part of your data
* [x] Construct one or multiple docker files for your code
* [ ] Build the docker files locally and make sure they work as intended
* [ ] Write one or multiple configurations files for your experiments
* [ ] Used Hydra to load the configurations and manage your hyperparameters
* [ ] When you have something that works somewhat, remember at some point to do some profiling and see if
      you can optimize your code
* [ ] Use Weights & Biases to log training progress and other important metrics/artifacts in your code. Additionally,
      consider running a hyperparameter optimization sweep.
* [ ] Use Pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code

### Week 2

* [x] Write unit tests related to the data part of your code
* [x] Write unit tests related to model construction and or model training
* [x] Calculate the coverage.
* [x] Get some continuous integration running on the github repository
* [x] Create a data storage in GCP Bucket for you data and preferable link this with your data version control setup
* [x] Create a trigger workflow for automatically building your docker images
* [x] Get your model training in GCP using either the Engine or Vertex AI
* [ ] Create a FastAPI application that can do inference using your model
* [ ] If applicable, consider deploying the model locally using torchserve
* [ ] Deploy your model in GCP using either Functions or Run as the backend

### Week 3

* [ ] Check how robust your model is towards data drifting
* [ ] Setup monitoring for the system telemetry of your deployed model
* [ ] Setup monitoring for the performance of your deployed model
* [ ] If applicable, play around with distributed data loading
* [ ] If applicable, play around with distributed model training
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed

### Additional

* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [x] Make sure all group members have a understanding about all parts of the project
* [x] Uploaded all your code to github

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

8

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

s174479, s175393, s183909, s183998

### Question 3
> **What framework did you choose to work with and did it help you complete the project?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

We used the framework Transformers, to train an NLP classifier. The goal of the classifier was to classify tweets whether the author is Donald Trump or a Russian troll account. From the framework we used the two functions `AutoTokenizer` and `AutoModelForSequenceClassification` to do respectively tokenization and classification. Both functions are based on the pre-trained model [bert-base-uncased](https://huggingface.co/bert-base-uncased) from Huggingface. This is an English base BERT model, trained on a large English corpus of raw text, hence without any human labeling. It's uncased, meaning the model sees no difference between uppercase and lowercase letters. The classification task in our project was done with two labels, whether the selected tweet came from Donald Trump or a Russian Troll.

## Coding environment

> In the following section we are interested in learning more about you local development environment.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development enviroment, one would have to run the following commands*
>
> Answer:
For a new team member to have a compatible environment, we start by informing them we are using python 3.9. We would ask them to create an new environment by using `conda create -n ‘enviorment name’ python=3.9`

Furthermore, there is a file called requirements which contains all the used libraries and dependencies for this project. Thus, we expect them download all the dependencies using `pip install -r requirements.txt`. 

We have our GitHub repository with all of our code, which all members will be working on their own forked version. For cloning the repository, they will call the function `git clone https://github.com/<username>/tweet_classification.git`

To obtain the data used for the project the command `dvc pull` will be used, which would download the raw and processed versions of the data.

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. Did you fill out every folder or only a subset?**
>
> Answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
> Answer:

We did use the cookiecutter template to structure our code. From the cookiecutter template we have used selected premade folders src, report and models. Then we have added a .dvc folder to handle data storage and version control. Furthermore we added a tests folders to include all the unit tests for our code. Besides that we customised the makefile and requirements.txt file. And the readme file was edited such that it contained a detailed description of our project. And as we plan to utilise google cloud platform we added 3 configuration files and two dockerfiles. Furthermore, we have added a .github/workflows folder for CI. 

### Question 6

> **Did you implement any rules for code quality and format? Additionally, explain with your own words why these**
> **concepts matters in larger projects.**
>
> Answer length: 50-100 words.
>
> Answer:

We implemented flake8 to check if each line of code contains max 100 characters and this is enforced using black. And we remove unused imports using autoflake and sorting the remaining imports in the optimal order using isort. Thereafter we also comment our code in a consistent manner for efficient code comprehension for other users. 

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

In total we have implemented two main tests. One for testing the data, and one for testing the model.

For the data test we are testing whether there are the same amount of tweets and labels, if the data are the right datatypes (tweets as strings, and labels as integers either 0 or 1), and if there are any NA's.

For the model we test the shape of the input and output of the model.

### Question 8

> **What is the total code coverage (in percentage) of your code? If you code had an code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

The total code coverage of our code is 83.4%. As mentioned above we have included tests for our `make_dataset.py` and our `model.py`. However, it is not guaranteed that the code is mostly error free as there could be errors in lines of code which have not been tested or not captured by the tests, such as if we only test for inpute type but not if there is an input. There are other python documents such as predict_model.py and train_model.py which we did not create tests for, thus these scripts are not tested. And this place further constrain on our code coverage, thereby reliability of how error free the source codes are.  

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

We made very much use branches and especially pull requests. To avoid excessive merge conflicts, all team members worked on their own forked version of the project. Such that every time we wanted to commit changes to the code, we had to do a pull request. But for the size of the project being quite small and we wanted to speed up the collaboration, we granted each member access to confirm his/hers own pull requests. When making changes to the code we create new branches so that it us to go back to previous working version if something breaks in the new updates. 

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

We attempted to make use of DVC, however it turned out to be unnecessary for our project since there are no changes in data over time and therefore no need for versioning. That being said, in the case where we would continuously require new tweets or data in general that would most likely lead to model decay over time, retraining is required. When that happens, we will definitely used DVC to keep track of data versions so we make sure to retrain the model on the updated dataset, as well as keeping previous version in case the retrained model fails. 

### Question 11

> **Discuss you continues integration setup. What kind of CI are you running (unittesting, linting, etc.)? Do you test**
> **multiple operating systems, python version etc. Do you make use of caching? Feel free to insert a link to one of**
> **your github actions workflow.**
>
> Answer length: 200-300 words.
>
> Example:
> *We have organized our CI into 3 separate files: one for doing ..., one for running ... testing and one for running*
> *... . In particular for our ..., we used ... .An example of a triggered workflow can be seen here: <weblink>*
>
> Answer:

We have organized our CI into 3 separate files: "Auto linter" for sorting imports using isort, "Auto linter flake8" for organizing linebreaks and removing unused imports, "Install with cache" for unit testing as mentioned earlier. We have selected some unit testing being optional, mainly regarding data preparation. We have selected testing on 3 operating systems, namely windows, mac and linux. We selected these three because they cover majority of operating systems that are being used. This also covers the systems used by our team members. The python version we have selected to test on is only 3.9, since versions <3.9 were found to result in many conflicts with dvc and versions > 3.9 resulted in other incompatibilities with multiple dependencies. However multiple pytorch versions are being tested for which are 1.11.9, 1.12.0 and 1.13.0. We have selected multiple pytorch versions to test for since we are using models which highly depend on the input being in torch format thus we found it very important that our code can run in multiple pytorch versions. For the whole project we are using a lot of different libraries making the requirements list is very long thus we have selected using caching to reduce redundant installation time and improve efficiency.

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: python my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

--- question 12 fill here ---

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

--- question 13 fill here ---

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

--- question 14 fill here ---

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments? Include how you would run your docker images and include a link to one of your docker files.**
>
> Answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

We build our docker images automatically using a trigger in Google Cloud everytime we pushed changes to our repository. We did though disabled this trigger at times, to not run out of ressources. Hence only building new docker images, when we found it necessary. 

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

--- question 16 fill here ---

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

We used the following services:
- Engine. To train our model on a virtual machine.
- Bucket. To store our data and models in the cloud.
- Cloud Build (with Triggers). To automatically create new images, when GutHub changes were pushed.
- Container Registry. To store our images.
- Vertex AI. To train our model.

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

--- question 18 fill here ---

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

<img width="616" alt="Skærmbillede 2023-01-15 kl  18 21 06" src="https://user-images.githubusercontent.com/117659231/212556612-8707625e-1113-4f6c-829f-453c63cd01e0.png">


### Question 20

> **Upload one image of your GCP container registry, such that we can see the different images that you have stored.**
> **You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

<img width="709" alt="Skærmbillede 2023-01-15 kl  18 21 56" src="https://user-images.githubusercontent.com/117659231/212556632-cca37eb7-adad-4f80-a7c6-892901cd169e.png">

### Question 21

> **Upload one image of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

<img width="998" alt="Skærmbillede 2023-01-15 kl  18 22 28" src="https://user-images.githubusercontent.com/117659231/212556651-602d0b6d-4f45-48bb-9cce-8685990e5d4e.png">

### Question 22

> **Did you manage to deploy your model, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

We first tried to get a FastAPI app up and running locally, but among others due to our bad performing computers we were not able to run the model locally. 
We then tried to deploy it with Cloud Functions, where we had a lot of issues with torch and loading the model correctly...

### Question 23

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

Nope not yet...
It could help in the long run to inform us about the behaviour of the model. Warn us if the data or the predictions started to become screwed for instance. 

### Question 24

> **How many credits did you end up using during the project and what service was most expensive?**
>
> Answer length: 25-100 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ...*
>
> Answer:

We ended up using XXX (30) credits during the project. 
The credits were spent on the following services:
- Cloud Build, 0.11 credits 
- Cloud Storage, 16.83 credits
- Compute Engine, 4.53 credits
- Networking, 0.17 credits
- Vertex AI, 0.22 credits
Hence clearly Cloud Storage has cost the most. 

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 25

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally in your own words, explain the**
> **overall steps in figure.**
>
> Answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and puch to github, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

XXX

### Question 26

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

We had several challenges.
One of the biggest challenges was our older slow computers with too little memory and RAM run the scripts. Luckily we had one good enough computer, where we could run stuff locally. Else we had to solve this challenge by uploading to the google cloud storage and testing there without having tested locally. This worked out but took significantly longer.
Another issue was that we didn't save our data correctly in the bucket. We tried using dvc to store the data in google cloud, but this did not work, hence we had to upload the data manually to our bucket.

### Question 27

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**
>
> Answer length: 50-200 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
>
> Answer:

--- question 27 fill here ---
