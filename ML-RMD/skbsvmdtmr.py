# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 19:51:27 2019

@author: Ranajoy
"""

# multi-class classification with mixed-Machine
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import csv
import mmconstants as c
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

datasetsvm = np.loadtxt(c.inputFilePath+c.inputFileNameForSVMData, delimiter=',')
rs,cs = datasetsvm.shape
M = datasetsvm[:,:cs-1]
n = datasetsvm[:,[cs-1]]

# Feature extraction
test = SelectKBest(score_func=f_classif, k=200)
fit = test.fit(M, n)

# Summarize scores
np.set_printoptions(precision=3)
print(fit.scores_)

features = fit.transform(M)
print(features)


dim=M.size
M_train, M_test, n_train, n_test = train_test_split(features, n, random_state = 0)
from sklearn.svm import SVC
svm_model_linear = SVC(kernel = c.linear, C = 1).fit(M_train, n_train)
svm_predictions = svm_model_linear.predict(M_test)
accuracy = svm_model_linear.score(M_test, n_test)
print(c.accInSVM,accuracy)
cm = confusion_matrix(n_test, svm_predictions)
with open(c.inputFileNameForDNNData,c.w) as f1:
    writer=csv.writer(f1, delimiter=',',lineterminator='\n',)
    for row_index, (input, prediction, label) in enumerate(zip (M_test, svm_predictions, n_test)):
     if prediction != label:
      result=np.append(input,label)
      writer.writerow(result)
      #print(c.incorrectPredictionData,input,label,prediction)

datasetdnn = np.loadtxt(c.inputFilePath+c.inputFileNameForDNNData, delimiter=',')
rd,cd = datasetdnn.shape
X = datasetdnn[:,:cd-1]
y = datasetdnn[:,[cd-1]]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

# training a DescisionTreeClassifier 

from sklearn.tree import DecisionTreeClassifier
dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train.ravel()) 
dtree_predictions = dtree_model.predict(X_test) 
acc = accuracy_score(y_test, dtree_predictions)
print("Accuracy in Decision Tree",acc)

#https://www.researchgate.net/post/Dataset_for_Multiclass_classification - For Glass Dataset


