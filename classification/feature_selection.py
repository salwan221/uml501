
#importing the libraries
import pandas as pd

#importing the dataset
dataset = pd.read_csv('classificationDataSet.csv')
x = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

#feature selector 1
from sklearn.feature_selection import SelectKBest
fs1=SelectKBest(k=5)
x_new1=fs1.fit_transform(x,y)

#feature selector 2
from sklearn.feature_selection import SelectFdr
fs2=SelectFdr()
x_new2=fs2.fit_transform(x,y)

#feature selector 3
from sklearn.linear_model import LogisticRegression
estimator = LogisticRegression()
from sklearn.feature_selection import RFE
fs3=RFE(estimator,5)
x_new3=fs3.fit_transform(x,y)

#feature selector 4
from sklearn.feature_selection import SelectFromModel
fs4=SelectFromModel(estimator)
x_new4=fs4.fit_transform(x,y)

#feature selector 5
from sklearn.feature_selection import SelectFwe
fs5=SelectFwe()
x_new5=fs5.fit_transform(x,y)


