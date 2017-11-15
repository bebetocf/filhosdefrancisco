import pandas as pd
import math

def sqr(x):
    return x*x

class BaysianClassifier:
    def __init__(self, train):
        self.d = len(train.columns)
        self.ammount = train.iloc[:,0].groupby('LABEL').count()
        self.total = self.ammount.sum()
        self.priori = self.ammount/self.total
        self.data_mean =  train.groupby('LABEL').mean()
        delta = train - self.data_mean
        self.variance = delta.apply(sqr).groupby('LABEL').mean().mean(axis = 1)
        
    def likelihood(self, label, x):
        var = self.variance[label]
        mean = self.data_mean.loc[label]
        
        power = -1.0 / (2 * var) * (x-mean).apply(sqr).mean()
        p = pow(2*math.pi * var , -self.d/2.0) * math.exp(power)  
        return p
    
    def posteriori(self, label, x):
        return self.priori.loc[label] * self.likelihood(label, x)
    
    def test(self, x, y):
        maxx = 0;
        for i in self.priori.index:
            p = self.posteriori(i,x)
            if(maxx < p):
                maxx = p
                label = i
        
        return label == y
    
    def test_label(self, x):
        maxx = 0
        for i in self.priori.index:
            p = self.posteriori(i,x)
            if(maxx < p):
                maxx = p
                label = i
        
        return label
    
    def test_df_label(self, test):
        v = []
        for i in range(len(test)):
            x = self.test_label(test.iloc[i])
            v += [x]
        return v
    
    
    def test_df(self, test):
        correct = 0
        for i in range(len(test)):
            if self.test(test.iloc[i], test.index[i]):
                correct +=1
        return correct
      
    