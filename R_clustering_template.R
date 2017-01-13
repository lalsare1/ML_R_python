#Clustering Template in R
#Importing the mall Data Set
dataset = read.csv('Mall_Customers.csv')
X = dataset[4:5]

# #Using the elbow method to find the optimal number of clusters
# set.seed(6)
# wcss = vector()
# for (i in 1:10) wcss[i] = sum(kmeans(X, i)$withinss)
# plot(1:10, wcss, type = 'b', main = paste('Cluster of Clients'),
#      xlab = "Number of Clusters", ylab = "WCSS")

#Finding optimal number of clusters using Dendograms
dendogram = hclust(dist(X, method = 'euclidean'), 
                   method = 'ward.D')
plot(dendogram,
     main = paste('Dendogram'),
     xlab = 'Customers',
     ylab = 'Euclidean Distance')

# #Applying k-means to the client dataset
# set.seed(29)
# k_means = kmeans(X, 5, iter.max = 1000, nstart = 10000)

#Fitting hierarchial clustering model to dataset
hc = hclust(dist(X, method = 'euclidean'), 
            method = 'ward.D')
y_hc = cutree(hc, 5)

#Visualizing the clusters
library(cluster)
clusplot(X,
         clus = y_hc,
         lines = 0,
         shade = TRUE,
         color = TRUE, 
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste('Clusters of clients'),
         xlab = "Annual Income",
         ylab = "Spending Score")