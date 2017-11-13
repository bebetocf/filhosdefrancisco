import pandas as pd
import numpy as np

data = pd.read_csv("segmentation_test.csv")

data_shape = pd.concat([data.iloc[:, 0:9], data.iloc[:, 19]], axis=1)
data_color = data.iloc[:, 9:20]

test = data


df = data.set_index('LABEL')
 
d = len(df.columns)

def test(df):
    ammount = df.iloc[:,0].groupby('LABEL').count()
    ammount.columns = ['count']
    for t in range(30):
        
        group = {}
        for i in ammount.index:
            group[i] = np.split(df.loc[i], 10)
        
        train = df
        test = pd.Dataframe
        for j in range(10):
            for i in ammount.index:
                temp = group[i].pop(0)
                test = pd.concat([test, temp])
                train = pd.concat([train] + group[i])
                group[i].append(temp)
                
                #test example 
                
        df = df.sample(frac=1)