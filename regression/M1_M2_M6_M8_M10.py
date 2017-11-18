
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

df=pd.DataFrame()

#fitting the linear model
from sklearn.linear_model import LinearRegression
regressor1=LinearRegression()
regressor1.fit(x_train,y_train)
y_pred1=regressor1.predict(x_test)
df['pred1']=y_pred1

#fitting the polynomial model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x_train)
regressor2=LinearRegression()
regressor2.fit(x_poly,y_train)
y_pred2=regressor2.predict(poly_reg.fit_transform(x_test))
df['pred2']=y_pred2

#fitting the bayesian ridge model
from sklearn.linear_model import BayesianRidge
regressor3=BayesianRidge()
regressor3.fit(x_train,y_train)
y_pred3=regressor3.predict(x_test)
df['pred3']=y_pred3

#fitting the k neighbors model
from sklearn.neighbors import KNeighborsRegressor
regressor4=KNeighborsRegressor(n_neighbors=3)
regressor4.fit(x_train,y_train)
y_pred4=regressor4.predict(x_test)
df['pred4']=y_pred4

#fitting the random forest model
from sklearn.ensemble import RandomForestRegressor
regressor5=RandomForestRegressor(n_estimators=500,random_state=0)
regressor5.fit(x_train,y_train)
y_pred5=regressor5.predict(x_test)
df['pred5']=y_pred5

#ensembling
y_pred=df.mean(axis=1)

#calculating r2
r2=r2_score(y_test,y_pred)

#calculating r
r=m.sqrt(r2)

#calculating error
error=mean_absolute_error(y_test,y_pred)

#calculating accuracy
accuracy = (float)(np.count_nonzero(np.array(abs(y_test - y_pred) <= 100))/np.size(y_test))*100

