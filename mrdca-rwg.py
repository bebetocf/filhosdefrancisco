import pandas as pd
import numpy as np

data = pd.read_csv("segmentation_test.csv")

data['CLUSTER'] = -1 * np.ones((data.shape[0], 1))

# data_shape = data.iloc[:, 0:9]
# data_color = data.iloc[:, 9:19]
# data_shape['CLUSTER'] = -1 * np.ones((data_shape.shape[0], 1))
# data_color['CLUSTER'] = -1 * np.ones((data_color.shape[0], 1))

shape_ini = 0
shape_end = 9
color_ini = 9
color_end = 19

# d_total = [];

K = 7
q = 3
p = 2
D = []

def random_prototypes(v, k, q):
    np.random.shuffle(v)
    v = v.reshape(10, len(v)/10)
    return v[:k,:q]

def calculate_dissimilarity(data, min_index, max_index, G):
    k = G.shape[0]
    q = G.shape[1]
    D = np.zeros((data.shape[0], k))
    for i in range(0, k):
        for j in range(0, q):
            diff = (data.iloc[:, min_index:max_index] - data.iloc[G[i][j], min_index:max_index])
            diff = (diff.apply(np.square).sum(axis = 1).apply(np.sqrt)).values
            D[:,i] += diff
    return D

def choose_cluster(data, G, D, lambda_):
    # global d_total
    D[0] = D[0] * lambda_[0];
    D[1] = D[1] * lambda_[1];
    d_total = D[0] + D[1]
    data['CLUSTER'] = d_total.argmin(axis = 1)

def best_prototypes(data, G, D, lambda_):



# G = np.zeros((K, q), dtype=np.int)
# D = calculate_dissimilarity(data, shape_ini, shape_end, G)

for it in range(0, 100):
    # Initializaiton
    # print it
    lambda_ = np.ones(p)
    G = random_prototypes(data.index.values, K, q)
    D[:] = []
    D.append(calculate_dissimilarity(data, shape_ini, shape_end, G))
    D.append(calculate_dissimilarity(data, color_ini, color_end, G))
    choose_cluster(data, G, D, lambda_)
    t = 0

    stop_calculate = false
    while not stop_calculate:
        t = t + 1
