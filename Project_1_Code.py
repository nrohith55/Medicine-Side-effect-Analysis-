# -*- coding: utf-8 -*-
"""
Created on Sat May 23 12:09:21 2020

@author: Rohith
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("E:\\Data Science\\Project\\Data\\train.csv")

df_train=df.iloc[:,[3,7]]

df1=pd.read_csv("E:\\Data Science\\Project\\Data\\test.csv")

df1_test=df.iloc[:,[3,7]]


# to clean the test and training data set:

import re
def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    i= re.sub("[\W+""]", " ",i)        
    w = []
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))

df_train['review']=df.review.apply(cleaning_text)
df1_test['review']=df.review.apply(cleaning_text)

#Now splitting the data into train and test:

trainX=df_train.review
trainy=df_train.output
testX=df1_test.review
testy=df1_test.output



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


tv = TfidfVectorizer()
trainX= tv.fit_transform(df_train.review)
testX = tv.fit_transform(df1_test.review)
y=df_train.output
from imblearn.over_sampling import SMOTE

sm=SMOTE(random_state=444)

X_train_res, y_train_res = sm.fit_resample(trainX, trainy) 
X_train_res.shape
y_train_res.shape
testX.shape
testy.shape

model=LogisticRegression()
model.fit(trainX,trainy)

y_pred=model.predict(testX)

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

print(accuracy_score(testy,y_pred))
print(classification_report(testy,y_pred))
print(confusion_matrix(testy,y_pred))

################.... using oversampling to reduce class imbalance..################################

model2=LogisticRegression()
model2.fit(X_train_res,y_train_res)

y_pred2=model2.predict(testX)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(testy,y_pred2))
print(classification_report(testy,y_pred2))
print(confusion_matrix(testy,y_pred2))

==========================================================================================================
from sklearn.tree import DecisionTreeClassifier

model3=DecisionTreeClassifier(criterion='entropy')
model3.fit(trainX,trainy)

y_pred3=model3.predict(testX)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(testy,y_pred3))
print(classification_report(testy,y_pred3))
print(confusion_matrix(testy,y_pred3))


################.... using oversampling to reduce class imbalance..##########################################

model4=DecisionTreeClassifier(criterion='entropy')
model4.fit(X_train_res,y_train_res)

y_pred4=model4.predict(testX)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(testy,y_pred4))
print(classification_report(testy,y_pred4))
print(confusion_matrix(testy,y_pred4))
================================================================================================================================

from sklearn.ensemble import RandomForestClassifier

model5=RandomForestClassifier(n_estimators=100)
model5.fit(trainX,trainy)

y_pred5=model5.predict(testX)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(testy,y_pred5))
print(classification_report(testy,y_pred5))
print(confusion_matrix(testy,y_pred5))

################.... using oversampling to reduce class imbalance..##########################################

model6=RandomForestClassifier(n_estimators=100)
model6.fit(X_train_res,y_train_res)

y_pred6=model6.predict(testX)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(testy,y_pred6))
print(classification_report(testy,y_pred6))
print(confusion_matrix(testy,y_pred6))

=========================================================================================================================================

from sklearn.naive_bayes import MultinomialNB

model7=MultinomialNB()
model7.fit(trainX,trainy)

y_pred7=model7.predict(testX)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(testy,y_pred7))
print(classification_report(testy,y_pred7))
print(confusion_matrix(testy,y_pred7))

################.... using oversampling to reduce class imbalance..##########################################

model8=MultinomialNB()
model8.fit(X_train_res,y_train_res)

y_pred8=model8.predict(testX)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(testy,y_pred8))
print(classification_report(testy,y_pred8))
print(confusion_matrix(testy,y_pred8))

=========================================================================================================================================

from sklearn.neural_network import MLPClassifier

model9=MLPClassifier(hidden_layer_sizes=(5,5))

model9.fit(trainX,trainy)

y_pred9=model9.predict(testX)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(testy,y_pred9))
print(classification_report(testy,y_pred9))
print(confusion_matrix(testy,y_pred9))

################.... using oversampling to reduce class imbalance..##########################################

model10=MLPClassifier(hidden_layer_sizes=(5,5))
model10.fit(X_train_res,y_train_res)

y_pred10=model10.predict(testX)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(testy,y_pred10))
print(classification_report(testy,y_pred10))
print(confusion_matrix(testy,y_pred10))
=========================================================================================================================================================










































