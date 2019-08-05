# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 09:49:28 2019

@author: KAVITA DWIVEDI
"""

#SupportVectormachine
  
  
from sklearn.svm import SVC
clf = SVC()

clf.fit(features_train,label_train)

pred_train = clf.predict(features_train)
pred_test = clf.predict(features_test)

    
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(label_test,pred_test))
print(classification_report(label_test,pred_test))
    