# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 21:19:43 2019

@author: Subhranil
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

fields = ['T_degC','Salnty']

# Taking only the first 500 values
dataset = pd.read_csv("Bottle_data.csv" , usecols = fields, dtype = {'T_degC' : float, 'Salnty' : float})
X = dataset.iloc[:500,0].values
y = dataset.iloc[:500,1].values

X = np.reshape(X,(-1,1))

# Removing Missing Values 
from sklearn.preprocessing import Imputer
imputer_X = Imputer(missing_values = 'NaN', strategy = 'mean' , axis = 0)
X = imputer_X.fit_transform(X)
imputer_y = Imputer(missing_values = 'NaN', strategy = 'mean' , axis = 0)
y = imputer_y.fit_transform(np.reshape(y,(-1,1)))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Building the Regressor
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# Predicting the Test Set Results
y_pred = regressor.predict(y_test)

# Visualising Training Set
plt.scatter(X_train,y_train, color = "red")
plt.plot(X_train, regressor.predict(X_train) , color = "blue")
plt.title("Oceanographic Data")
plt.xlabel("Temperature in degree Celcius")
plt.ylabel("Salinity")

# Visualising the Test Set
plt.scatter(X_test,y_test, color = "red")
plt.plot(X_test, regressor.predict(X_test) , color = "blue")
plt.title("Oceanographic Data")
plt.xlabel("Temperature in degree Celcius")
plt.ylabel("Salinity")