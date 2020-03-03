import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

df = pd.read_csv('iris.csv', header=None)
df = df.replace(' ?', np.nan)
df.dropna(inplace=True)
df = df.sample(frac=1).reset_index(drop=True)

labels = {}
enu = 0
for i in set(list(df[4])):
    labels[i] = enu
    enu+=1

data = np.array(df)

data_dict = {}

for i in range(len(data)):
    data[i,4] = labels[data[i,4]]
    data_dict[str(data[i,:4])] = data[i,4]

def cmeans(x, c=4, m=2, iterations=10):
    data = x
    x = scale(data[:,:2])
    u = np.random.rand(x.shape[0], c)
    epsilon = 0.000001
    centers= np.zeros(shape=(c, x.shape[1]))
    for it in range(iterations):
        u_prev = u
        #update c
        for j in range(len(centers)):
            temp1=0
            temp2=0
            for i in range(len(x)):
                temp1 += u[i,j]**m * x[i]
                temp2 += u[i,j]**m
            centers[j] = temp1/temp2
        #update u

        for i in range(len(x)):
            for j in range(len(centers)):
                temp = 0
                for k in range(len(centers)):
                    temp += (np.linalg.norm(x[i] - centers[j])/np.linalg.norm(x[i]-centers[k]))**2/(m-1)
                    u[i][j] = 1/temp

        #if np.linalg.norm(u-u_prev) < epsilon:
        #    break
    clusters = {}
    for i in range(c):
        clusters[i] = []
    preds = []
    for sample in range(len(u)):
        p = np.argmax(u[sample])
        preds.append(p)
        clusters[p].append(data[sample])
    preds=np.array(preds)

    return u, centers, clusters, preds


data = data.astype('float64')
features = scale(data[:,:2])
num_centroids=3
u, centers, clusters, preds = cmeans(data, c=3)


colors= ['r', 'g', 'y']

plt.scatter(features[preds==0,0], features[preds==0,1], color=colors[0])
plt.scatter(features[preds==1,0], features[preds==1,1], color=colors[1])
plt.scatter(features[preds==2,0], features[preds==2,1], color=colors[2])
plt.scatter(centers[:,0],centers[:,1], color='b', s=100, label='centroids')

plt.show()