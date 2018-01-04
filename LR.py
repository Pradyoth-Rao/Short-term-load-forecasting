#Implments the linear regression model. Accuracy of 40%

import pandas as pd
import csv, math , datetime
import time
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression, LogisticRegression
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
#All the libraries required
df = pd.read_csv("loaddata.txt")
df = df[['Date','Hour','Minute','Volume']]
#our data frame contains the follwoing we can choose what we want
forcast_col='Volume'
#the column that we want to predict
df.fillna(-99999, inplace=True)
#print(df)
forecast_out = int(math.ceil(0.02*len(df)))
#this is the logic we need to know, how many days prediction can be calculated in this eqaution.
df['label'] = df[forcast_col].shift(-forecast_out)
#print(df)
#print(forecast_out)

"""
df = pd.read_csv("loaddata.txt")
df = df[['Date','Hour','Minute','Volume']]
#our data frame contains the follwoing we can choose what we want
forcast_col='Volume'
#the column that we want to predict
df.fillna(-99999, inplace=True)
#print(df)
forecast_out = int(math.ceil(0.02*len(df)))
#this is the logic we need to know, how many days prediction can be calculated in this eqaution.
df['label'] = df[forcast_col].shift(-forecast_out)
#print(df)
#print(forecast_out)
"""

X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
#This contains all the data till the forecasy_out value
X = X[:-forecast_out:]
df.dropna(inplace=True)
#print(df)
df.dropna(inplace=True)
y = np.array(df['label'])
y = np.array(df['label'])
#print(len(X),len(y))
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)
#the size of the testing data
clf = LinearRegression()
clf.fit(X_train,y_train)
accuracy = clf.score(X_test,y_test)
forecast_set = clf.predict(X_lately)
print('Linear Regression accuracy:', accuracy*100,'%')
df['Volume'].plot()
df['label'].plot()
plt.legend(loc=4)
plt.xlabel('Records')
plt.ylabel('Volume in MW')
plt.show()
 


