# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 13:59:29 2021

@author: wlom
"""

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset=pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values
# print(x)
# print(y)

#Training the Linear Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x,y)

#training the polynomial regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
pr=PolynomialFeatures(degree=4) #degree means the eqn will run upto b4*x4
x_poly=pr.fit_transform(x)     #also means it will be more more accurate but in this case degree=4 will cause overfitting
lr_2=LinearRegression()
lr_2.fit(x_poly,y)

#Visualising the Linear Regression results
plt.scatter(x,y, color='red')
plt.plot(x,lr.predict(x),color='blue')
plt.title('Position vs Salary(linear reg)')
plt.xlabel('postion level')
plt.ylabel('salary')
plt.show()


#visualizing the Polynomial regression result
plt.scatter(x, y, color = 'red')
plt.plot(x, lr_2.predict(pr.fit_transform(x)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Visualising the Polynomial Regression results (for higher resolution and smoother curve)
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, lr_2.predict(pr.fit_transform(x_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#linear reg prediction
print(lr.predict([[6.5]]))

#poly reg prediction
print(lr_2.predict(pr.fit_transform([[6.5]])))