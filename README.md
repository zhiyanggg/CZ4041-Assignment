# Allstate Claims Severity

Allstate is one of the largest insurance company in the United States, and they develop automated methods of predicting claims severity in order to improve their claims service for the over 16 million households they protect. The goal of the challenge is to explore a variety of machine learning methods and choose a regression to create an algorithm which best accurately predicts claims severity.

The dataset includes 130 features, out of which 116 are categorical and 14 are real-valued features. There are 188,318 training examples and there is a loss associated with each training example.

In this readme, an algorithm to determine the severity of a claim in terms of its cost given a dataset of previous claims will be explored.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

RStudio

### After installing

First, you will need to install the following libraries in R:

library(mlr)
library(xgboost)
library(data.table)
library(parallelMap)
library(FeatureHashing)
library(BBmisc)
library(Matrix)
library(glmnet)

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

## Authors

* **Lim Zhi Yang** 

Refer to the report for who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
