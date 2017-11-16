import pandas as pd
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score

data = pd.read_csv("segmentation_test.csv")

data['CLUSTER'] = -1 * np.ones((data.shape[0], 1))
data['LABEL'] = data['LABEL'].astype('category')
data['LABEL'] = data['LABEL'].astype('category').cat.codes

# data_shape = data.iloc[:, 0:9]
# data_color = data.iloc[:, 9:19]
# data_shape['CLUSTER'] = -1 * np.ones((data_shape.shape[0], 1))
# data_color['CLUSTER'] = -1 * np.ones((data_color.shape[0], 1))

shape_ini = 0
shape_end = 9
color_ini = 9
color_end = 19

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

def add_dissimilarity(D, lambda_):
    D[0] = D[0] * lambda_[0];
    D[1] = D[1] * lambda_[1];
    return D[0] + D[1]

def choose_cluster(data, G, D, lambda_):
    d_total = add_dissimilarity(D, lambda_)
    newClusters = d_total.argmin(axis = 1)
    if(np.array_equal(data['CLUSTER'],newClusters)):
        return True
    else:
        data['CLUSTER'] = newClusters
        return False


# def best_prototypes(data, G, D, lambda_):
#     d_total = add_dissimilarity(D, lambda_)

def best_weight(data, G, D, lambda_):
    K = G.shape[0]
    q = G.shape[1]
    p = len(D)
    denom = []

    for h in range(0, p):
        data_cluster = 0
        for k in range(0, K):
            data_cluster += (D[h][data['CLUSTER'] == k][:,k]).sum(axis = 0)
        denom.append(data_cluster)

    num = reduce(lambda x, y: x*y, denom)
    for j in range(0, p):
        lambda_[j] = num/denom[j]

def calculate_J(D, lambda_):
    ret = 0
    for k in range(0, K):
        for h in range(0, p):
            ret += lambda_[h] * (D[h][data['CLUSTER'] == k][:,k]).sum(axis = 0)
    return ret

for it in range(0, 1):
    # Initializaiton
    # print it
    lambda_ = np.ones(p)
    G = random_prototypes(data.index.values, K, q)
    D[:] = []
    D.append(calculate_dissimilarity(data, shape_ini, shape_end, G))
    D.append(calculate_dissimilarity(data, color_ini, color_end, G))
    choose_cluster(data, G, D, lambda_)
    t = 0

    best_G = np.copy(G)
    best_cluster = (data.iloc[:,19:21]).copy()
    best_lambda = np.copy(lambda_)
    best_J = float("inf")

    stop_calculate = False
    while not stop_calculate:
        t = t + 1
        # print t
        # TODO: Calculate best prototypes

        # Recalculate dissimilarity matrix
        D[:] = []
        D.append(calculate_dissimilarity(data, shape_ini, shape_end, G))
        D.append(calculate_dissimilarity(data, color_ini, color_end, G))

        # Recalculating weight vector
        best_weight(data, G, D, lambda_)

        # Choose new cluster for the objects
        stop_calculate = choose_cluster(data, G, D, lambda_)

        print calculate_J(D, lambda_)

    J = calculate_J(D, lambda_)
    if(J < best_J):
        best_G = np.copy(G)
        best_cluster = (data.iloc[:,19:21]).copy()
        best_lambda = np.copy(lambda_)
        best_J = J

CR = adjusted_rand_score(best_cluster['LABEL'], best_cluster['CLUSTER'])
print CR