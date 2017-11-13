import pandas as pd
import math

data = pd.read_csv("segmentation_test.csv")

data_shape = pd.concat([data.iloc[:, 0:9], data.iloc[:, 19]], axis=1)
data_color = data.iloc[:, 9:20]

test = data

ammount = data.iloc[:,18:20].groupby('LABEL').count()
ammount.columns = ['count']
total = ammount.sum()
priori = ammount/total


df = data.set_index('LABEL')
data_mean = df.groupby('LABEL').mean()
 
delta = df - data_mean

def sqr(x):
    return x*x

delta_squared = delta.apply(sqr)

variance = delta_squared.groupby('LABEL').mean().mean(axis=1)


d = len(df.columns)

def likelihood(label, x):
    var = variance[label]
    mean = data_mean.loc[label]
    
    power = -1.0 / (2 * var) * (x-mean).apply(sqr).mean()
    p = pow(2*math.pi * var , -d/2.0) * math.exp(power)  
    return p

def posteriori(label, x):
    priori.loc[label]*likelihood(label, x)
    
def baysian_test(index):
    sample = test.iloc[index]
    y = sample['LABEL']
    x = sample.iloc[:d]
    maxx = 0;
    for i in priori.index:
        p = posteriori(i,x)
        if(maxx < p):
            maxx = p
            label = i
    
    return label == y
        
    