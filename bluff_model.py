#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 14:29:41 2019

@author: siddharth.m98@gmail.com
"""
# An Machine learning model that can help HR team by detecting the bluff of an candidate regarding the negotiation salary wrt the years of experience.

# First we need to import 3 libraries for this. These are numpy,matplotlib(if you want to visualize) and pandas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../input/Salary_Data.csv')
X = dataset.iloc[:, 0:1].values  # We use this python function to get the 1st coloumn alone as it is X
y = dataset.iloc[:, 1].values # We use this to get the 2nd coloumn alone which is the one we want to predict Y

# Be any regression model, we need to use linear regression to fit out data into the model so we must import LinearRegression from the sklearn.linear_model class
from sklearn.linear_model import LinearRegression 
lin_reg  = LinearRegression()
lin_reg.fit(X,y)

#Till now we can use this model for making a linear prediction. BUt since we want super accuracy, we must use polynomial prediction. where we have to give powers multiplication in linear manner. 
#So we use our PolynomialFeatures from sklearn.preprocessing class for this. Later we transform the current X data to polynomial data using lin_reg.

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5) # we use the degree as 5 here for best prediction.
X_poly = poly_reg.fit_transform(X)

#We perform fitting of our polynomial data here using lin_reg2.
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)

#We can use matplotlib for using plt.scatter to get the visual data.

plt.scatter(X,y,color = 'red')
plt.plot(X,lin_reg.predict(X),color = 'blue')
plt.title('truth or bluff - linear reg')
plt.xlabel('Experience level')
plt.ylabel('salary')
plt.show()



plt.scatter(X,y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color = 'blue')
plt.title('truth or bluff - linear reg')
plt.xlabel('Experience level')
plt.ylabel('salary')
plt.show()

# Now we can predict any salary by passing years of experience as parameter.

lin_reg2.predict(poly_reg.fit_transform([[6.5]]))
