# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 18:55:03 2017

@author: vpsiddhant
"""
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

datasetva = pd.read_csv('147.csv')
X = datasetva.iloc[:, [1,2, 3]].values

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 6, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
fig = plt.figure()

#ax = fig.add_subplot(111, projection='3d')
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=40, azim=134)
ax.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], X[y_kmeans == 0, 2], s = 100, c = 'red', label = 'Cluster 1')
ax.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1],X[y_kmeans == 1, 2], s = 100, c = 'blue', label = 'Cluster 2')
ax.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1],X[y_kmeans == 2, 2], s = 100, c = 'green', label = 'Cluster 3')
ax.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1],X[y_kmeans == 3, 2], s = 100, c = 'cyan', label = 'Cluster 4')
ax.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1],X[y_kmeans == 4, 2], s = 100, c = 'magenta', label = 'Cluster 5')
ax.scatter(X[y_kmeans == 5, 0], X[y_kmeans == 5, 1],X[y_kmeans == 5, 2], s = 100, c = 'black', label = 'Cluster 5')
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],kmeans.cluster_centers_[:, 2], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of gears')
ax.set_xlabel('Speed')
ax.set_ylabel('RPM')
ax.set_zlabel('Speed')
#ax.legend()
plt.show()

#cOUNTING HOW MANY DATA POINTS IN EACH CLUSTER
a = [0,0,0,0,0,0]


for i in range(0,len(y_kmeans)):
    if y_kmeans[i] == 0:
            a[0] = a[0] + 1
    elif y_kmeans[i] == 1:
            a[1] = a[1] + 1
    elif y_kmeans[i] == 2:
            a[2] = a[2] + 1
    elif y_kmeans[i] == 3:
            a[3] = a[3] + 1
    elif y_kmeans[i] == 4:
            a[4] = a[4] + 1
    elif y_kmeans[i] == 5:
            a[5] = a[5] + 1

print(a)      

# Silhouette analysis
silhouette_avg = silhouette_score(X, y_kmeans)

print(silhouette_avg)

datasetva['CLUSTER'] = pd.Series(y_kmeans, index=datasetva.index)

#datasetva.to_csv('ouput.csv',index=False)