import pandas as pd
import math



def knn(train, x, y, k):
    #print(train.index.name)
    neigbords = (train - x).apply(abs).sum(axis = 1).nsmallest(k, 'last').groupby('LABEL').count()
    
    maxx = 0
    for i in neigbords.index:
        if(neigbords[i] > maxx):
            label = i   
            maxx = neigbords[i]
    return label == y 
    
def knn_label(train, x, k):
    
    neigbords = (train - x).apply(abs).sum(axis = 1).nsmallest(k, 'last').groupby('LABEL').count()
    
    maxx = 0
    for i in neigbords.index:
        if(neigbords[i] > maxx):
            label = i   
            maxx = neigbords[i]
    return label 

def knn_lables(test, train, k):
    v = []
    for i in range(len(test)):
        v += [knn_label(test.iloc[i], test, k)]
    
    return v

def knn_dataframes(test, train, k):
    correct = 0
    for i in range(len(test)):
        if knn(train, test.iloc[i], test.index[i], k):
            correct += 1
    return correct
