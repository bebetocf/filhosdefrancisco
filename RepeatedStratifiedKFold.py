# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

data = pd.read_csv("~/Documents/vbas/test.csv")

shape_view = pd.concat([data.iloc[:, 0:9], data.iloc[:, 19]], axis=1)
color_view = data.iloc[:, 9:20]
classes = data["LABEL"]
classes_index = pd.factorize(classes)

X = data.iloc[:, 0:9].values
y = classes_index[0]
print classes_index[1]


kf = RepeatedStratifiedKFold(n_splits=10, n_repeats=30)
count = 0
for train_index, test_index in kf.split(X, y):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    count +=1
