**PCA.cpp**
  
  **Model**: Principal Component Analysis

  **Algorithm**: Dimensionality Reduction

  **Steps**: сenter data, cov matrix compute, eigen decomposition, projection


**Description**

Implemented PCA for unsupervised dimensionality reduction.

The algorithm steps:
  **Data Centering**
    
    m_j = (1 / N) * sum x_{ij}(where x_{ij} is the j-th feature of the i-th sample, sum is from 1 to N)
    Center the data:
    x_centered = x - m

  **Covariance Matrix Computation**
    
    S[i][j] = (1 / (N - 1)) * sum (x_k[i] * x_k[j]) (where S captures how the features vary together, sum is from 1 to N)

  **Eigen Decomposition**
    
    S * v = lambda * v
    Obtain eigenvalues - l and eigenvectors - v, where l represent the variance explained by each component.
    Sort eigenvalues and eigenvectors in descending order.

  
  **Projection**
    
    Select top k eigenvectors to form a projection matrix V_k.
    Project the centered data onto the new basis:
    X_projected = X_centered * V_k
