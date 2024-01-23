# Prediction of Credit Card Fraud Detection

## Overview:
This project aims to develop a machine learning model for detecting fraudulent credit card transactions using a dataset of transactions made by European cardholders in September 2013. The dataset is highly imbalanced, with only a small percentage of transactions labeled as fraud.

## Dataset:
The dataset contains transactions made in two days, with 492 frauds out of 284,807 transactions. The positive class (frauds) accounts for 0.172% of all transactions. The dataset file is named creditcard.csv(can be downloaded from [kagle.com](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud))

## Resources and Dependencies:
#### Python:
Jupyter Notebook (version 6.5.4)
#### Dependencies:
pandas

numpy

seaborn

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

## Project Details:
#### Exploratory Data Analysis:
Checked for null values/missing values in the dataset. Checked the imbalance in the class in the dataset. Descriptive Statistics done for valid and Fraud transactions.How the fraud and the Valid occurs with respect to the transaction amount and distribution of the amount for positive(frauds) and negative(Valid) is shown in a graph. Distribution of time also analysed as it doesnot largely affect the dataset.

#### Data Preprocessing :
Under sampling has been done as data is highly unbalanced hence sample dataset has been created containing similar distribution of valid and fraud transation that is 492 transaction each.

#### Split the data into features and targets.

#### Split the data into Training data and Test data

#### Model Training :
The machine learning model used for this project is Logistic Regression. Training data is used to train the logistic regression model and model evaluation done.

#### Result:
Result achieved by model using test dataset:
Accuracy score of Test data: 0.8983739837398373

## Conclusion:
As the data was highly imbalanced hence under sampling is used. Logistic Regression model gives around 90 percent accuracy in both Test data as well as training data.
