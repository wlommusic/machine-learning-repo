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