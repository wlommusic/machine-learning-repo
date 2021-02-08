# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 11:56:19 2021

@author: wlom
"""

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset

data = pd.read_csv('Salary_Data.csv')
x= data.iloc[:,:-1].values  #rows,col
y= data.iloc[:,-1].values

#splitting the data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

#training the linear model on training set

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)

#predicting the test results
y_pred=reg.predict(x_test)
# print(y_pred)

#visuallizing the training set results
plt.scatter(x_train,y_train ,color='blue')
plt.plot(x_train,reg.predict(x_train),color='red')
plt.title('Salary vs Exp (test set)')
plt.xlabel('Years of Exp')
plt.ylabel('Salary')
plt.show()

#visualizing the test set results
plt.scatter(x_test,y_test ,color='blue')
plt.plot(x_train,reg.predict(x_train),color='red')
plt.title('Salary vs Exp (test set)')
plt.xlabel('Years of Exp')
plt.ylabel('salary')
plt.show()