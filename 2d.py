import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_samples, silhouette_score

datasetva = pd.read_csv('147.csv')
X = datasetva.iloc[:, [1, 2]].values

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 6, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(X[y_kmeans == 5, 0], X[y_kmeans == 5, 1], s = 100, c = 'black', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of gears')
plt.xlabel('Speed')
plt.ylabel('Engine Load')
plt.legend()
plt.show()

# Silhouette analysis
silhouette_avg = silhouette_score(X, y_kmeans)

print(silhouette_avg)

# Ouput to CSV
datasetva['CLUSTER'] = pd.Series(y_kmeans, index=datasetva.index)

datasetva.to_csv('ouput.csv',index=False)