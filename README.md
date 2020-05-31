# Predicting Customer Lifetime Value

We will use past purchase history of customers to build a model that can predict the Customer Lifetime Value (CLV) for new customers

## Loading and Viewing Data

We will load the data file and checkout summary statistics and columns for that file.

```
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics

raw_data = pd.read_csv("history.csv")

raw_data.dtypes
```

The dataset consists of the customer ID, the amount the customer spent on your website for the first months of his relationship with your business and his ultimate life time value ( say 3 years worth)

```
raw_data.head()
```

## Do Correlation Analysis

```
cleaned_data = raw_data.drop("CUST_ID",axis=1)
cleaned_data .corr()['CLV']
```

We can see that the months do show strong correlation to the target variable (CLV). That should give us confidence that we can build a strong model to predict the CLV

## Do Training and Testing Split

Let us split the data into training and testing datasets in the ratio 90:10.

```
predictors = cleaned_data.drop("CLV",axis=1)
targets = cleaned_data.CLV
pred_train, pred_test, tar_train, tar_test  =   train_test_split(predictors, targets, test_size=.1)
print( "Predictor - Training : ", pred_train.shape, "Predictor - Testing : ", pred_test.shape )
```
Predictor - Training :  (90, 6) Predictor - Testing :  (10, 6)

## Build and Test Model

We build a Linear Regression equation for predicting CLV and then check its accuracy by predicting against the test dataset

```
# Build model on training data
model = LinearRegression()
model.fit(pred_train,tar_train)
print("Coefficients: \n", model.coef_)
print("Intercept:", model.intercept_)

# Test on testing data
predictions = model.predict(pred_test)
predictions

sklearn.metrics.r2_score(tar_test, predictions)
```
It shows a 91% accuracy. This is an excellent model for predicting CLV

## Predicting for a new Customer

Let us say we have a new customer who in his first 3 months have spend 100,0,50 on your website. Let us use the model to predict his CLV.

```
new_data = np.array([100,0,50,0,0,0]).reshape(1, -1)
new_pred=model.predict(new_data) 
print("The CLV for the new customer is : $",new_pred[0])
```
The CLV for the new customer is : $ 4018.59836236

# kk289-Predicting-Customer-Lifetime-Value-CLV-
# kk289-Predicting-Customer-Lifetime-Value-CLV-
