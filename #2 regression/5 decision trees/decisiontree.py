# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 21:30:12 2021

@author: wlom
"""

#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#train the model
from sklearn.tree import DecisionTreeRegressor
DR=DecisionTreeRegressor()
DR.fit(x,y)

#predicting the values
y_pred = DR.predict([[6.5]])
print(f'predicted salary: {y_pred}')


#visualizing the results
x_grid=np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color='purple')
plt.plot(x_grid,DR.predict(x_grid),color='orange')
plt.show()

# the decision tree is pretty bad at predicting values
# with just one variable its more designed 
# for datasets with multiple variables
# thats why in this cas its pretty inaccurate