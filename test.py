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
    
    result = pd.DataFrame(columns=['knn_shape','knn_color','baysian_shape','baysian_color','majority'])
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
            #print(test_shape.columns)
            #print(test_shape.index.name)
            #print(test_color.columns)
            #print(test_color.index.name)
            #print("shalala")
            knn_shape_ans = knn.knn_lables(test_shape, train_shape, 3)
            knn_color_ans = knn.knn_lables(test_color, train_color, 3)
            baysian_shape_ans = BaysianClassifier(train_shape).test_df_label(test_shape)
            baysian_color_ans = BaysianClassifier(train_color).test_df_label(test_color)
            
            
            majority = []
            for i in range(len(knn_shape_ans)):
                ans = []
                ans.append(knn_shape_ans[i])
                ans.append(knn_color_ans[i])
                ans.append(baysian_shape_ans[i])
                ans.append(baysian_color_ans[i])
                x = pd.DataFrame({'label':ans,'count':[1,1,1,1]})
                #print(x)
                majority += [x.groupby('label').count().nlargest(1,'count').index[0]]
            
            knn_shape_correct = knn.knn_dataframes(test_shape, train_shape, 3)
            knn_color_correct = knn.knn_dataframes(test_color, train_color, 3)
            baysian_shape_correct = BaysianClassifier(train_shape).test_df(test_shape)
            baysian_color_correct = BaysianClassifier(train_color).test_df(test_color)
            majority_correct = sum(comp(majority , test_shape.index))
            
            result.loc[index] = [knn_shape_correct, knn_color_correct, baysian_shape_correct, baysian_color_correct, majority_correct]
            index += 1;
        out_str = "out_"+str(t)+".csv"
        result.to_csv(out_str)
        df = df.sample(frac=1)

test(df)