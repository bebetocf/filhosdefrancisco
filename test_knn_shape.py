import pandas as pd
import numpy as np
import knn
from BaysianClassifier import BaysianClassifier
data = pd.read_csv("segmentation_test.csv")



test = data


df = data.set_index('LABEL')
 
d = len(df.columns)



def compare(a,b):
    if(a == b): return 1
    return 0

comp = np.vectorize(compare)

def test(df):
    
    
    ammount = df.iloc[:,0].groupby('LABEL').count()
    ammount.columns = ['count']
    
    result = pd.DataFrame(columns=['knn_shape1','knn_shape3','knn_shape5','knn_shape7','knn_shape9'])
    index = 0
    for t in range(30):
        print("test:"+str(t))
        df_shape = df.iloc[:, 0:9]
        df_color = df.iloc[:, 9:20]
        #print (df_shape)
        #print (df_color)
        group_shape = {}
        group_color = {}
        for i in ammount.index:
            group_shape[i] = np.split(df_shape.loc[i], 10)
            group_color[i] = np.split(df_color.loc[i], 10)
        
        
        
        
        for j in range(10):
            train_shape = pd.DataFrame(columns = df_shape.columns)
            test_shape = pd.DataFrame(columns = df_shape.columns)
            
            train_color = pd.DataFrame(columns = df_color.columns)
            test_color = pd.DataFrame(columns = df_color.columns)
            print("j:" +str(j))
            for i in ammount.index:
                temp = group_shape[i].pop(0)
                test_shape = pd.concat([test_shape, temp])
                train_shape = pd.concat([train_shape] + group_shape[i])
                group_shape[i].append(temp)
                
                temp = group_color[i].pop(0)
                test_color = pd.concat([test_color, temp])
                train_color = pd.concat([train_color] + group_color[i])
                group_color[i].append(temp)
            
            test_shape.index.name = "LABEL"
            test_color.index.name = "LABEL"
            train_shape.index.name = "LABEL"
            train_color.index.name = "LABEL"
            
            knn_shape_correct = knn.knn_dataframes(test_shape, train_shape, 1)
            knn_color_correct = knn.knn_dataframes(test_shape, train_shape, 3)
            baysian_shape_correct = knn.knn_dataframes(test_shape, train_shape, 5)
            baysian_color_correct = knn.knn_dataframes(test_shape, train_shape, 7)
            majority_correct = knn.knn_dataframes(test_shape, train_shape, 9)
            
            result.loc[index] = [knn_shape_correct, knn_color_correct, baysian_shape_correct, baysian_color_correct, majority_correct]
            
            index += 1;
        print(result.mean()/210.0)
        df = df.sample(frac=1)
    result.mean().to_csv("knn_shape.csv")
test(df)