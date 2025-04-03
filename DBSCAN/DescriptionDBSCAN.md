**DBSCAN.cpp**

  **Model**: Density-Based Spatial Clustering of Applications with Noise (DBSCAN)


**Description**

Implemented DBSCAN for unsupervised clustering.

The algorithm is computed:

    1. For each point x in the dataset, compute its ε-neighborhood:
       N_eps(x) = { y ∈ X | distance(x, y) < eps } 
       (where distance(x, y) is computed using the Euclidean metric.)

    2. A point x is considered a core point if:
       |N_eps(x)| ≥ minPts
       (where |N_eps(x)| is the number of points within the eps-neighborhood of x.)

    3. For each unvisited point:
       - If the point is not a core point, label it as noise (-1).
       - Otherwise, if the point is a core point, create a new cluster and expand the cluster by
         recursively adding all points that are density-reachable from x.
         
    4. The cluster expansion is performed by the function expandCluster, which updates the cluster assignment for
       all density-reachable points.
       
    5. Points that are not included in any cluster after the entire process are labeled as noise.
