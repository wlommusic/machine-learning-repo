# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 21:51:43 2021

@author: wlom
"""
#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing the dataset
data = pd.read_csv('Data.csv')
x= data.iloc[:,:-1].values  #rows,col
y= data.iloc[:,-1].values

print(x)
print(y)

# taking care of missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])


print(x)

#encoding categorical data
#encoding independent values
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import  OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
x = np.array(ct.fit_transform(x)) 
print(x)

#encoding dependent values
from sklearn.preprocessing import  LabelEncoder
le = LabelEncoder()
y=le.fit_transform(y)
print(y)

#splitting dataset into training and test set
from sklearn.model_selection import  train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
print(x_train)
print(x_test)
print(y_train)
print(y_test)

#feature scaling
from sklearn.preprocessing import  StandardScaler
sc=StandardScaler()
x_train[:,3:]=sc.fit_transform(x_train[:,3:])
x_test[:,3:]=sc.fit_transform(x_test[:,3:])
print(x_train)
print(x_test)