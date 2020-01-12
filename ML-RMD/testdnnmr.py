# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 18:19:56 2020

@author: Ranajoy
"""

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

datasetdnn = np.loadtxt(c.inputFilePath+c.inputFileNameForDNNData, delimiter=',')
rd,cd = datasetdnn.shape
X = datasetdnn[:,:cd-1]
Y = datasetdnn[:,[cd-1]]
dummy_y = np_utils.to_categorical(Y)
model = Sequential()
model.add(Dense(13, input_dim=cd-1, activation=c.relu))
model.add(Dense(9, activation='relu'))
model.add(Dense(5, activation='sigmoid'))
model.compile(loss=c.categorical_crossentropy, optimizer=c.adam, metrics=[c.acu])
model.fit(X, dummy_y, epochs=150, batch_size=10)
_, accuracy = model.evaluate(X, dummy_y)
print("DNNAccuracy",accuracy)