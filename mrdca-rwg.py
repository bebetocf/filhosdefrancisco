import pandas as pd
import numpy as np
import math
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import preprocessing
from tqdm import tqdm
from datetime import datetime

def normalizeDataframe(df):
    min_max_scaler = preprocessing.MinMaxScaler()
    data_scaled = pd.DataFrame(min_max_scaler.fit_transform(data.loc[:, data.columns != 'LABEL']))
    data_scaled['LABEL'] = data['LABEL']
    data_scaled.columns = data.columns
    return data_scaled

data = pd.read_csv("segmentation_test.csv")

# Nomarlize Features
data = normalizeDataframe(data)

data['CLUSTER'] = -1 * np.ones((data.shape[0], 1))
data['LABEL'] = data['LABEL'].astype('category')
data['LABEL'] = data['LABEL'].astype('category').cat.codes

file_name = "mrdca-rwg_" + str(datetime.now().strftime("%Y-%m-%d_%H:%M")) + ".out"

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

def calculate_dissimilarity(data, min_index, max_index):
    n = data.shape[0]
    D = np.zeros((n, n))
    for i in range(0, n):
        diff = (data.iloc[:, min_index:max_index] - data.iloc[i, min_index:max_index])
        diff = (diff.apply(np.square).sum(axis = 1).apply(np.sqrt)).values
        D[:,i] += diff
    return D

def add_dissimilarity(D, lambda_):
    ret = []
    ret.append(D[0] * lambda_[0])
    ret.append(D[1] * lambda_[1])
    return ret[0] + ret[1]

def choose_cluster(data, G, D, lambda_):
    d_total = add_dissimilarity(D, lambda_)
    n = data.shape[0]
    K = G.shape[0]
    q = G.shape[1]
    d_cluster = np.zeros((n, K))
    for i in range(0,K):
        for j in range(0,q):
            d_cluster[:, i] += (d_total[:,G[i][j]])

    newClusters = d_cluster.argmin(axis = 1)
    if(np.array_equal(data['CLUSTER'],newClusters)):
        return True
    else:
        data['CLUSTER'] = newClusters
        return False


def best_prototypes(data, G, D, lambda_):
    n = data.shape[0]
    K = G.shape[0]
    d_total = add_dissimilarity(D, lambda_)

    for k in range(0, K):
        d_prototype = np.zeros(n)
        for i in range(0, n):
            temp = d_total[data['CLUSTER'] == k]
            d_prototype[i] = (temp[:,i].sum())
        d_prototype = np.argsort(d_prototype)
        G[k,:] = d_prototype[0:3]

def best_weight(data, G, D, lambda_):
    n = data.shape[0]
    K = G.shape[0]
    q = G.shape[1]
    p = len(D)
    denom = []

    d_cluster = []
    for h in range(0,p):
        use = D[h]
        temp = np.zeros((n, K))
        for i in range(0,K):
            for j in range(0,q):
                temp[:, i] += (use[:,G[i][j]])
        d_cluster.append(temp)

    for h in range(0, p):
        data_cluster = 0
        for k in range(0, K):
            data_cluster += (d_cluster[h][data['CLUSTER'] == k][:,k]).sum(axis = 0)
        denom.append(data_cluster)

    num = reduce(lambda x, y: x*y, denom)
    num = math.sqrt(num)
    for j in range(0, p):
        lambda_[j] = num/denom[j]

def calculate_J(D, lambda_, G):
    n = D[0].shape[0]
    K = G.shape[0]
    q = G.shape[1]
    ret = 0
    d_total = add_dissimilarity(D, lambda_)

    temp = np.zeros((n, K))
    for i in range(0,K):
        for j in range(0,q):
            temp[:, i] += (d_total[:,G[i][j]])

    for k in range(0, K):
        ret += (temp[data['CLUSTER'] == k][:,k]).sum(axis = 0)
    return ret

D[:] = []
D.append(calculate_dissimilarity(data, shape_ini, shape_end))
D.append(calculate_dissimilarity(data, color_ini, color_end))

best_J = float("inf")
best_G = np.copy((K, q))
best_cluster = (data.iloc[:,19:21]).copy()
best_lambda = np.copy(np.ones(p))

for it in tqdm(range(0, 200)):
    file = open(file_name, "a")

    # Initializaiton
    file.write("Repeticao: " + str(it))
    file.write("\n")
    # print ("Repeticao: " + str(it))
    lambda_ = np.ones(p)
    G = random_prototypes(data.index.values, K, q)
    choose_cluster(data, G, D, lambda_)
    t = 0


    file.write(("\tJ (init): " + str(calculate_J(D, lambda_, G))))
    file.write("\n")
    # print(("\tJ (init): " + str(calculate_J(D, lambda_, G))))

    stop_calculate = False
    while not stop_calculate:
        t = t + 1
        file.write("\tIteracao: " + str(t))
        file.write("\n")
        # print("\tIteracao: " + str(t))

        # Calculate best prototypes
        best_prototypes(data, G, D, lambda_)
        file.write(("\t\tJ (prototypes): " + str(calculate_J(D, lambda_, G))))
        file.write("\n")
        # print (("\tJ (prototypes): " + str(calculate_J(D, lambda_, G))))

        # Recalculating weight vector
        best_weight(data, G, D, lambda_)
        file.write(("\t\tJ (weight): " + str(calculate_J(D, lambda_, G))))
        file.write("\n")
        # print (("\tJ (weight): " + str(calculate_J(D, lambda_, G))))

        # Choose new cluster for the objects
        stop_calculate = choose_cluster(data, G, D, lambda_)
        file.write(("\t\tJ (organize): " + str(calculate_J(D, lambda_, G))))
        file.write("\n")
        # print (("\tJ (organize): " + str(calculate_J(D, lambda_, G))))


    J = calculate_J(D, lambda_, G)
    if(J < best_J):
        best_G = np.copy(G)
        best_cluster = (data.iloc[:,19:21]).copy()
        best_lambda = np.copy(lambda_)
        best_J = J

    file.write("\n")
    # print("\n")

    CR = adjusted_rand_score(data['LABEL'], data['CLUSTER'])
    file.write("\tCR: " + str(CR) + "\n")
    file.write("\n\n")
    # print(str(CR))
    # print("\n\n")

    file.close()



file = open(file_name, "a")

CR = adjusted_rand_score(best_cluster['LABEL'], best_cluster['CLUSTER'])
file.write("Best result:\n")
file.write("\tCR: " + str(CR) + "\n")
file.write("\tShape weight: " + str(best_lambda[0]) + "\n")
file.write("\tColor weight: " + str(best_lambda[1]) + "\n")
file.write("\tClusters prototypes:\n\t")
for j in range(0, q):
    file.write("\t\t" + str(j) + "\t")
file.write("\n")

for i in range(0, K):
    file.write("\t\t" + str(i+1))
    for j in range(0, q):
        file.write("\t" + str(best_G[i][j]) + "\t")
    file.write("\n")


# print("Best CR: " + str(CR))
file.close()
