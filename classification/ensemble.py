
#importing the libraries
import pandas as pd
from sklearn.metrics import confusion_matrix

#importing the dataset
dataset = pd.read_csv('classificationDataSet.csv')
x = dataset.iloc[:,1:].values
y = dataset.iloc[:, 0].values

#splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/5)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

df=pd.DataFrame()

#fitting the logistic regression model on the training test
from sklearn.linear_model import LogisticRegression
classifier1 = LogisticRegression(max_iter=10000)
classifier1.fit(x_train, y_train)
y_pred1 = classifier1.predict(x_test)
df['pred1']=y_pred1

#fitting the model on the training test
from sklearn.linear_model import SGDClassifier
classifier2 = SGDClassifier()
classifier2.fit(x_train, y_train)
y_pred2 = classifier2.predict(x_test)
df['pred2']=y_pred2

#fitting the gaussian process model on the training set
from sklearn.gaussian_process import GaussianProcessClassifier
classifier3 = GaussianProcessClassifier()
classifier3.fit(x_train, y_train)
y_pred3 = classifier3.predict(x_test)
df['pred3']=y_pred3

#fitting the model on the training test
from sklearn.naive_bayes import GaussianNB
classifier4=GaussianNB()
classifier4.fit(x_train, y_train)
y_pred4 = classifier4.predict(x_test)
df['pred4']=y_pred4

#fitting the model on the training test
from sklearn.gaussian_process import GaussianProcessClassifier
classifier5 = GaussianProcessClassifier()
classifier5.fit(x_train, y_train)
y_pred5 = classifier5.predict(x_test)
df['pred5']=y_pred5

#fitting the model on the training test
from sklearn.neighbors import KNeighborsClassifier
classifier6=KNeighborsClassifier(n_neighbors=100)
classifier6.fit(x_train, y_train)
y_pred6 = classifier6.predict(x_test)
df['pred6']=y_pred6

#fitting the model on the training test
from sklearn.neighbors import RadiusNeighborsClassifier
classifier7= RadiusNeighborsClassifier(radius=10)
classifier7.fit(x_train, y_train)
y_pred7 = classifier7.predict(x_test)
df['pred7']=y_pred7

#fitting the model on the training test
from sklearn.neighbors import NearestCentroid
classifier8 = NearestCentroid()
classifier8.fit(x_train, y_train)
y_pred8 = classifier8.predict(x_test)
df['pred8']=y_pred8

#fitting the model on the training test
from sklearn.tree import DecisionTreeClassifier
classifier9=DecisionTreeClassifier(criterion='entropy')
classifier9.fit(x_train, y_train)
y_pred9 = classifier9.predict(x_test)
df['pred9']=y_pred9

#fitting the model on the training test
from sklearn.ensemble import RandomForestClassifier
classifier10=RandomForestClassifier(n_estimators=500,criterion='entropy')
classifier10.fit(x_train, y_train)
y_pred10 = classifier10.predict(x_test)
df['pred10']=y_pred10

#ensembling
df1 = pd.DataFrame()
df2 = pd.DataFrame()
df3 = pd.DataFrame()
df4 = pd.DataFrame()
df5 = pd.DataFrame()

df1 = df[['pred1','pred5','pred6','pred7','pred10']].copy()
df2 = df[['pred1','pred2','pred4']].copy()
df3 = df[['pred2','pred4','pred6','pred8','pred10']].copy()
df4 = df[['pred5','pred7','pred8']].copy()
df5 = df[['pred1','pred2','pred6','pred8','pred10']].copy()

test=df5
comp=len(test.columns)
y_pred=(test==1).astype(int).sum(axis=1)/comp > 0.5
y_pred=y_pred.astype(int)

#confusion matrix
cm=confusion_matrix(y_test,y_pred)
tn=cm[0][0]
fp=cm[0][1]
fn=cm[1][0]
tp=cm[1][1]

#sensitiviry
sensitivity=(tp)/(tp+fn)

#specificity
specificity=(tn)/(tn+fp)

#precision
precision=(tp)/(tp+fp)

#recall
recall=(tp)/(tp+fn)

#accuracy
accuracy=(tn+tp)/(tp+fn+fp+tn)