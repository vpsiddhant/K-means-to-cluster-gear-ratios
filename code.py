# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 09:57:14 2017

@author: vpsiddhant
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import preprocessing


def remove_outliers(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    indices = []
    for i in range(0,len(arr)):
        if np.abs((arr[i] - mean))/std > 3:
            indices.append(i)
            
    return indices

def check_speed(arr):
    print("inside function")
    indices = []
    #arr = arr.tolist()
    for i in range(0,len(arr)):
        if arr[i] is not 0:
            continue
            #print("inside")
        else:
            print("inside")
            indices.append(i)
    return indices
    
#Getting dataset
datasetva = pd.read_csv('147.csv')

#Removing outliers and zero speed

speed = datasetva.iloc[:, [1]].values
rpm = datasetva.iloc[:, [2]].values
engineload = datasetva.iloc[:, [3]].values

speed_indices = remove_outliers(speed)
rpm_indices = remove_outliers(rpm)
engine_indices = remove_outliers(engineload)
#zero_speed_indices = check_speed(speed)

zero_speed_indices = np.where(speed == 0)[0]

all_indices = []

for i in range(0, len(rpm_indices)):
    all_indices.append(rpm_indices[i])
    
for i in range(0, len(speed_indices)):
    if speed_indices[i] not in all_indices:
        all_indices.append(speed_indices[i])
        
for i in range(0, len(engine_indices)):
    if engine_indices[i] not in all_indices:
        all_indices.append(engine_indices[i])

for i in range(0, len(zero_speed_indices)):
    if zero_speed_indices[i] not in all_indices:
        all_indices.append(zero_speed_indices[i])
        
print(len(datasetva))
        
#for i in range(0,len(all_indices)):
 #   datasetva2 = datasetva.drop(datasetva.index[all_indices[i]])
datasetva2 = datasetva.drop(datasetva.index[all_indices])
#print(len(datasetva2))

print(len(datasetva2))

#Normalizing data
X = datasetva2.iloc[:, [1,2]].values

n = len(X)
listofzeroes = [0]*n
datasetva2['gear_ratio'] = pd.Series(listofzeroes, index=datasetva2.index)

X = datasetva2.iloc[:, [1,2,4]].values

X = preprocessing.normalize(X, norm='l2')

#finding gear ratios
XVA = X.tolist()

for i in range(0,n):
        XVA[i][2] = float(XVA[i][1])/XVA[i][0]

X = np.asarray(XVA)

#KMEANS cluster
kmeans = KMeans(n_clusters = 6, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)


#Number of points in each cluster
a = [0,0,0,0,0,0]
for i in range(0,len(y_kmeans)):
    a[y_kmeans[i]] = a[y_kmeans[i]] + 1

print(a)

#Finding average gear ratio for each cluster

datasetva2['Gear ratios'] = pd.Series(X[:,2], index=datasetva2.index)
datasetva2['CLUSTER'] = pd.Series(y_kmeans, index=datasetva2.index)
datasetva2.drop('gear_ratio', axis=1, inplace=True)
gear = datasetva2.iloc[:, [4,5]].values

gear = gear.tolist()

b = [0,0,0,0,0,0]
for i in range(0,len(gear)):
    b[int(gear[i][1])] = b[int(gear[i][1])] + gear[i][0]

avg_gear_ratio = [b[0]/a[0],b[1]/a[1],b[2]/a[2],b[3]/a[3],b[4]/a[4],b[5]/a[5]]


#Calculating silhoutte score
silhouette_avg = silhouette_score(X, y_kmeans)

print(silhouette_avg)


#printing to text file
datasetva2.to_csv('ouput_followup.csv',index=False)

f = open('avg.txt','w')

for i in range(0,6):
    f.write('The average gear ratio for cluster ' + repr(i) +' is ' + repr(avg_gear_ratio[i]) + '\n')

f.write('The average silhouette score is ' + repr(silhouette_avg) + '.' )
f.close()