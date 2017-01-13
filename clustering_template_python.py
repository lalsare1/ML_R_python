# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 02:07:33 2017

@author: AmoolyaD
"""

#Clustering Template for Python
#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

"""#Using elbow method to find optimal number of clusters for k-means clustering
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans =    KMeans(n_clusters= i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()"""

#Using Dendograms for finding the optimal number of clusters using Hierarchial Clustering
#Using dedograms to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidian Distances')
plt.show()

"""#Applying K-means to the dataset
kmeans = KMeans(n_clusters=5, init = 'k-means++', max_iter=300, n_init=10, random_state=0)
y_clust = kmeans.fit_predict(X)"""

#Fitting the hierarchial clustering model to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage= 'ward')
y_clust = hc.fit_predict(X)

# Visualizing the clusters
plt.scatter(X[y_clust==0, 0], X[y_clust==0, 1], s = 100, c = 'red', label = 'Careful' )
plt.scatter(X[y_clust==1, 0], X[y_clust==1, 1], s = 100, c = 'blue', label = 'Standard' )
plt.scatter(X[y_clust==2, 0], X[y_clust==2, 1], s = 100, c = 'green', label = 'Target' )
plt.scatter(X[y_clust==3, 0], X[y_clust==3, 1], s = 100, c = 'cyan', label = 'Careless' )
plt.scatter(X[y_clust==4, 0], X[y_clust==4, 1], s = 100, c = 'magenta', label = 'Sensible' )
plt.title('Cluster of clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()