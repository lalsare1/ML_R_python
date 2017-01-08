# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 20:10:17 2017

@author: AmoolyaD
"""

#SVR
#Data Preprocessing
#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

#Splitting the dataset into training set and test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
"""
#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)


#Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, Y)


#Predicting a new result with polynomial regression
y_pred = sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))


#Visaulizing the Regression Results
plt.scatter(X, Y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth of Bluff(SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')

"""#Visaulizing the Regression Results for higher resolution
X_grid = np.arrange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth of Bluff')
plt.xlabel('Position Level')
plt.ylabel('Salary')"""
