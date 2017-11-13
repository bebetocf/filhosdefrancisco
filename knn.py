import pandas as pd
import math

data = pd.read_csv("segmentation_test.csv")

data_shape = pd.concat([data.iloc[:, 0:9], data.iloc[:, 19]], axis=1)
data_color = data.iloc[:, 9:20]

test = data


df = data.set_index('LABEL')
 
d = len(df.columns)

def knn(train, x, y, k):
    
    neigbords = (train - x).apply(abs).sum(axis = 1).nsmallest(k, 'last').groupby('LABEL').count()
    
    maxx = 0    
    for i in neigbords.index:
        if(neigbords[i] > maxx):
            label = i   
            maxx = neigbords[i]
    return label == y 
    
    
test_data = data.set_index('LABEL')
train_data = data.set_index('LABEL')


def knn_dataframes(test, train, k):
    correct = 0
    for i in range(len(test)):
        if knn(train, test.iloc[i], test.index[i], k):
            correct +=1
    return correct

print(knn_dataframes(train_data,test_data,5)) 