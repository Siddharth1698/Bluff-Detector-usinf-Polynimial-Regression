#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 14:29:41 2019

@author: siddharth.m98@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 1].values


from sklearn.linear_model import LinearRegression
lin_reg  = LinearRegression()
lin_reg.fit(X,y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)
X_poly = poly_reg.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)

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

lin_reg2.predict(poly_reg.fit_transform([[6.5]]))