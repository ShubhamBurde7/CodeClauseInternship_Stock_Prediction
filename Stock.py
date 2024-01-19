import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import quandl
# It is the famous website which is basically used to exract the data which is related to economic and financial background.

data = pd.read_csv("E:/Internships/Codeclause Data Science/NSE-TATAGLOBAL11.csv")
# data = quandl.get("NSE/TATAGLOBAL")
print(data.head(10))

# Aim of model
# 1st - to find the closing value (Regression)
# 2nd - whether I need to sell(-1) or buy(+1) the stock? (Classification)

plt.figure(figsize=(12,6))      # figsize is used to bigger the figure size
plt.plot(data['Close'], label = 'Closing Price')
# plt.show()

data['Open - Close'] = data['Open'] - data['Close']
data['High - Low'] = data['High'] - data['Low']
data = data.dropna()            # used to drop the null value from the data
# Here we are taking the difference as the input features to predict whether the customer should buy or sell the stock

X = data[['Open - Close', 'High - Low']]
print(X.head())

Y = np.where(data['Close'].shift(-1)>data['Close'],1,-1)
# shift function the column up or down
# as the closing price of the stock is more than the previous day then the value of Y is set to be 1 
print(Y)

# Training and Testing part

#splitting the data into training and testing part
from sklearn.model_selection import train_test_split    # model_selection is library
X_train, X_test,y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=44)

# Implementation of KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

#using gridsearch to find the best parameter
params = {'n_neighbors': [2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
knn = neighbors.KNeighborsClassifier()
model = GridSearchCV(knn, params, cv = 5)
# K is value which represent the minimum number of neighbors I want to see whenever I am predicting the data
# To find the Optimal value of K, it depends upon the dataset we have
# also, we can use the approach called GridSearchCV
# here, K value is a Hyperparameter

# fit the model
model.fit(X_train, y_train)

# Accuracy Score
accuracy_train = accuracy_score(y_train, model.predict(X_train))
accuracy_test = accuracy_score(y_test, model.predict(X_test))

print('Train_data Accuracy: %.2f' %accuracy_train)
print('Test_data Accuracy: %.2f' %accuracy_test)

predictions_classification = model.predict(X_test)

actual_predicted_date = pd.DataFrame({'Actual Class': y_test, 'Predicted Class': predictions_classification})
print(actual_predicted_date.head(10))

y = data['Close']
print(y)

# Implementation of KNN Regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn import neighbors

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, Y, test_size=0.25, random_state=44)

#using gridsearch to find the best parameter
params = {'n_neighbors': [2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
knn_reg = neighbors.KNeighborsRegressor()
model_reg = GridSearchCV(knn_reg, params, cv = 5)

# fit the model and make the predictions
model_reg.fit(X_train_reg, y_train_reg)
predictions = model_reg.predict(X_test_reg)

print(predictions)

# rmse
rms = np.sqrt(np.mean(np.power((np.array(y_test)-np.array(predictions)),2)))
print(rms)

valid = pd.DataFrame({'Actual Close': y_test_reg, 'Predicted Close Value': predictions})
print(valid.head(10))