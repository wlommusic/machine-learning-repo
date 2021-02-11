# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 13:48:45 2021

@author: wlom
"""

#importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
y=y.reshape(len(y),1)

#feature scaling

from sklearn.preprocessing import StandardScaler
sc_x= StandardScaler()
sc_y= StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)
# print(x)
# print(y)

#training the svr model
from sklearn.svm import SVR
reg=SVR(kernel='rbf')
reg.fit(x,y)

#predicting the results

print(sc_y.inverse_transform(reg.predict(sc_x.transform([[6.5]]))))

#Visualising the SVR results
plt.scatter(sc_x.inverse_transform(x),sc_y.inverse_transform(y),color='blue')
plt.plot(sc_x.inverse_transform(x),sc_y.inverse_transform(reg.predict(x)),color='red')
plt.title('position vs salary (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')

plt.show()

#high res graph
x_grid = np.arange(min(sc_x.inverse_transform(x)), max(sc_x.inverse_transform(x)), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color = 'red')
plt.plot(x_grid, sc_y.inverse_transform(reg.predict(sc_x.transform(x_grid))), color = 'blue')
plt.title('position vs salary (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


