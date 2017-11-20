
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

#fitting the model on the training test
from sklearn.svm import SVC
classifier=SVC(max_iter=10000,kernel='rbf')
classifier.fit(x_train, y_train)

#predicting the test set results
y_pred = classifier.predict(x_test)

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