#Hierarchial Clustering

#Importing the mall Data Set
dataset = read.csv('Mall_Customers.csv')
X = dataset[4:5]

#Finding optimal number of clusters using Dendograms
dendogram = hclust(dist(X, method = 'euclidean'), 
                   method = 'ward.D')
plot(dendogram,
     main = paste('Dendogram'),
     xlab = 'Customers',
     ylab = 'Euclidean Distance')

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