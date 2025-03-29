**KNearestNeighbours**
   **model**: k - Nearest Neighbours



**Description**

Implemented KNearestNeighbours for binary classification

k-NN is a model that classifies a new data point based on the classes of the k closest points in the training set. 
The distance between points is computed using the Euclidean distance:

    d(x, y) = sqrt(sum (xᵢ - yᵢ)²)

For a given test sample, the algorithm:

    Computes the distance to each training sample.
    Selects the k closest neighbors.
    Determines the predicted class by majority vote among these neighbors.
