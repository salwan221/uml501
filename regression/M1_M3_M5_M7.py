
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

#fitting the lasso model
from sklearn.linear_model import Lasso
regressor2=Lasso(max_iter=10000)    
regressor2.fit(x_train,y_train)
y_pred2=regressor2.predict(x_test)
df['pred2']=y_pred2

#fitting the ridge model
from sklearn.linear_model import Ridge
regressor3=Ridge()
regressor3.fit(x_train,y_train)
y_pred3=regressor3.predict(x_test)
df['pred2']=y_pred3

#fitting the kernel ridge model
from sklearn.kernel_ridge import KernelRidge
regressor4=KernelRidge()
regressor4.fit(x_train,y_train)
y_pred4=regressor4.predict(x_test)
df['pred4']=y_pred4

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

