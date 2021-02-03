# Disaster Response Pipeline Project

The project uses the response data from various users across the world posted in social media during various disasters such as earthquake, fire, missing person, food/ water related issues and so on. The goal is to develop a Web App which will categorize any given disaster message received from people. T

## Table of Contents

- [Installation](#installation)
- [Overview of repository](#overview-of-repository)
- [How to run the code](#how-to-run-the-code)

## Installation

To clone the repository use: ``` git clone https://github.com/hianjana/Diasater_Response_Pipeline_Project_Udacity.git ```

## Overview of repository

The repository contains the following files:

    Diasater_Response_Pipeline_Project_Udacity/
    ├── README.md
    ├── ETL_Workflow.ipynb
    ├── ML_workflow.ipynb
    ├── data/
      ├── disaster_categories.csv
      ├── disaster_messages.csv
      ├── DisasterResponse.db
      ├── process_data.py
    ├── models/
      ├── train_classifier.py
      ├── classifier.pkl
    ├── app/
      ├── run.py
      ├── templates/
        ├── master.html
        ├── go.html

The input files used for the project are:

1) disaster_messages.csv - Responses from people in social media for various disasters.
2) disaster_categories.csv - Category under which each disaster response falls.

## How to run the code

There are 3 parts to this project:
1) ETL pipeline
2) Machine Learning pipeline
3) Webapp

**To run the ETL pipeline**
    ``` python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db ```
This will create a SQL database in the name DisasterResponse.db under the "data" folder.

**To run the ML pipeline**
    ``` python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl ```
This will create a classification model pickle file in the name classifier.pkl under the "models" folder.

**To run the Web App**
Go to the app folder and run:
    ``` python run.py ```
In a new terminal run:
    ``` env|grep WORK ```
The above command will give a SPACEID. Use the SPACEID and run the below url in a new browser.
    https://<Your SPACEID>-3001.udacity-student-workspaces.com/

The WebApp will be up and running now.
    
## High level data preparation steps taken

**ETL Pipeline**
1) Extract the data from the CSV files from the "data" folder.
2) Split the 'categories' column into individual categories.
3) For each category, fill the values with 0s and 1s.
4) Merge the two datasets.
5) Remove duplicates.
6) Load into a SQL database and save it in the "data" folder.

**ML Pipleline**
1) Load the data from the SQL database presnet in the "data" folder.
2) Perform data visualization to see the number of responses received per category.
3) Split the dataset into train and test.
4) Apply feature extraction techniques: CountVectorizer, TfidfTransformer.
5) Build a model using SGDClassifier.
6) Hyperparameter tune teh model using GridSearchCV.
7) Using the best estimator obtained, train the dataset.
8) Predict on test dataset.
9) Check the performance metrics of the model.
10) Train another model (AdaBoostClassifier) with another feature added using StartVerbExtractor (This step is performed only in the ipynb).
11) Save the trained model as a pickle file in the "models" folder.

**Web App*

For data visualization, 2 plots are created.
1) The top 10 categories: the ones with the highest number of responses

![alt text](https://github.com/hianjana/Diasater_Response_Pipeline_Project_Udacity/blob/main/plot1_top10.JPG)

2) The bottom 10 categories: the ones with the least number of responses

![alt text](https://github.com/hianjana/Diasater_Response_Pipeline_Project_Udacity/blob/main/plot2_bottom10.JPG)

