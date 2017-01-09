# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 00:52:07 2017

@author: AmoolyaD
"""

#Random Forest Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')

#Create a Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X, Y)




#Predicting a new result with polynomial regression
y_pred = regressor.predict(6.5)

#Visaulizing the Random Forest Regression Results for higher resolution
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth of Bluff')
plt.xlabel('Position Level')
plt.ylabel('Salary')