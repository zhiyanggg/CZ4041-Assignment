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
```

library(mlr)

library(xgboost)

library(data.table)

library(parallelMap)

library(FeatureHashing)

library(BBmisc)

library(Matrix)

library(glmnet)
```

## Running the tests

In order to see how well our solution fare, we will be comparing it to other regression algorithms. 
It is not mandatory to run the codes from the "Extra Experiments" section onwards as it is just for comparison purposes and will not
affect our proposed solution. 

Feel free to uncomment the code if you wish to view the comparison results.

### End to end tests

In the script, a few algorithms are tested. Regression algorithms such as:

1) XGBoost

2) Lasso Regression

3) H20 Deep Learning Model

4) H20 Gradient Boosting Machine

5) Linear Regression

can be compared and contrasted by running the script, "mlassignment.R".

From the start to the section on "Extra Experiments", the codes are based on the algorithm XGBoost. 
Following that, there is a section on Lasso Regression, H20 Deep Learning Model, H20 Gradient Boosting Machine and Linear Regression.

Do note that, the algorithms can be run independently without each other as the code logic do not overlap. Hence, you can choose 
not to run all algorithms if you wish to save time. In fact, you can test those selected algorithms that you prefer.

For XGBoost:

In the middle of the code, there is a portion of code which have the comments being numbered. By following the code logic, Section 1 to 4
are being used to find the optimal parameters for training the model. Cross validation is being used here to do the hyperparameter tuning.

To save time from finding the best parameters, the hyperparameters have been found and implemented in the code. This explains why
obj_par =1.79 in section 5. 

To get a better understanding of where the codes are, you can search for the following comment as one of the examples to find out where 
the numbered-comments codes are. 

```
Find section 1, whereby the comment is:
# 1) Define the set of parameters you want to tune (here we use only 'obj_par')
```

## Authors

* **Lim Zhi Yang** 

Refer to the contributors in this repository to see who participated in this project.
