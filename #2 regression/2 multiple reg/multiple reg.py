# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 19:47:39 2021

@author: wlom
"""
#importing the libraries
import numpy as np
import pandas as pd

#importing the dataset
dataset=pd.read_csv('50_Startups.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
#print(x)

#encoding the categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import  OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')
x=np.array(ct.fit_transform(x))

#print(x)

#splitting the dataset into the training and test sets
from sklearn.model_selection import  train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2,random_state=0)

#training the model
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)

#predicting the model

y_pred=reg.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))


# making a single prediction
"""
Making a single prediction 
(for example the profit of a startup with R&D Spend = 160000, Administration Spend = 130000, Marketing Spend = 300000 and State = 'California')
"""
print('the 1,0,0 is the code for california')
print(reg.predict([[1, 0, 0, 160000, 130000, 300000]]))

#Getting the final linear regression equation with the values of the coefficients
print(reg.coef_)
print(reg.intercept_)

#Profit=86.6×Dummy State 1−873×Dummy State 2+786×Dummy State 3−0.773×R&D Spend+0.0329×Administration+0.0366×Marketing Spend+42467.53