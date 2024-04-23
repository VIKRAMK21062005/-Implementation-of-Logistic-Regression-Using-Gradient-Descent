# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Load the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary. 6.Define a function to predict the 
   Regression value. 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: VIKRAM K
RegisterNumber:  212222040180
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("C:\sem-1\Placement_Data.csv")
dataset

dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop("salary",axis=1)
dataset ["gender"] = dataset ["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset ["hsc_b"].astype('category')
dataset ["degree_t"] = dataset ["degree_t"].astype('category')
dataset ["workex"] = dataset ["workex"].astype('category')
dataset["specialisation"] = dataset ["specialisation"].astype('category')
dataset ["status"] = dataset["status"].astype('category')
dataset ["hsc_s"] = dataset ["hsc_s"].astype('category')
dataset.dtypes


dataset ["gender"] = dataset ["gender"].cat.codes
dataset ["ssc_b"] = dataset["ssc_b"].cat.codes
dataset ["hsc_b"] = dataset ["hsc_b"].cat.codes
dataset ["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset ["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset

X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, -1].values
Y

theta=np.random.randn(X.shape[1])
y=Y

def sigmoid(z):
    return 1 / (1+np.exp(-z))

def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

def gradient_descent (theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y) / m
        theta -= alpha * gradient
    return theta

theta =  gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)

def predict(theta, X): 
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred
y_pred = predict(theta, X)
accuracy = np.mean(y_pred.flatten()==y)
print("Accuracy:", accuracy)
print(y_pred)
print(Y)
xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]]) 
y_prednew = predict(theta, xnew) 
print(y_prednew)
xnew = np.array([[0, 0, 0, 0, 0, 2, 8, 2, 0, 0, 1, 0]]) 
y_prednew = predict(theta, xnew) 
print(y_prednew)

## OUTPUT

![image](https://github.com/VIKRAMK21062005/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120624033/bf5fc379-ad36-40a1-8ef9-5a9782c9d7ce)

![image](https://github.com/VIKRAMK21062005/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120624033/a47bc4bb-10bf-476a-b3b8-3cf06b3c1d20)

![image](https://github.com/VIKRAMK21062005/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120624033/b9d3f008-2aaa-46f2-a165-1d82ef793aa7)

![image](https://github.com/VIKRAMK21062005/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120624033/3980e813-c11a-4c15-95be-7c4426a21c7b)
## y_pred

![image](https://github.com/VIKRAMK21062005/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120624033/518bbb4a-14cb-4452-be78-b69db741f8c8)
## Y:

![image](https://github.com/VIKRAMK21062005/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120624033/67fa0979-228c-45b3-a5f2-d968878bc53e)

## y_prednew

![image](https://github.com/VIKRAMK21062005/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/120624033/864cc353-c9e5-4cde-ba98-49cc0ff2eb8c)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

