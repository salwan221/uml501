
#importing the libraries
import pandas as pd
import math as m
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

#importing the dataset
dataset = pd.read_csv('regressionDataSet.csv')
x = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

#splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/5)

#fitting the model on the training set
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x_train)
regressor=LinearRegression()
regressor.fit(x_poly,y_train)

#predicting the test set results
y_pred=regressor.predict(poly_reg.fit_transform(x_test))

#calculating r2
r2=r2_score(y_test,y_pred)

#calculating r
r=m.sqrt(r2)

#calculating error
error=mean_absolute_error(y_test,y_pred)

#calculating accuracy
accuracy = (float)(np.count_nonzero(np.array(abs(y_test - y_pred) <= 100))/np.size(y_test))*100

