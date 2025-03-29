**KMeans.cpp**
   **model**: KMeans
   **loss function**: sum of squared distances
   **optimization**: iterative centroid update
   

**Description**

  Implemented k-Means Clustering for unsupervised learning.

  k-Means partitions the data into k clusters by following these steps:

    Randomly initialize k centroids from the training samples.
    For each sample, compute the Euclidean distance to each centroid:
      d(x, c) = sqrt(sum(xᵢ - cᵢ)²)
    Assign each sample to the nearest centroid.
    Update each centroid as the mean of all samples assigned to it.
    Repeat the assignment and update steps for a fixed number of iterations (or until convergence).

This approach minimizes the within-cluster sum of squared distances and finds the natural groupings in the data.
 
