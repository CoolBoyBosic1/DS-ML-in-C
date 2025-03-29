**DecisionTree.cpp**

   **Model**: Decision Tree
   
   **Loss function**: entropy, djini
   
   **Optimization**: greedy recursive partitioning


**Description**

  Implemented Decision Tree for classification.

  A Decision Tree is a model that recursively splits the dataset based on feature values.
  
  Entropy is a loss function which is computed as
  
    Entropy(S) = -sum( p(c) * log₂(p(c)) ) (where p(c) is the proportion of samples in S belonging to class c)

  Djini is a loss function which is computed as

    Gini(S) = 1 - sum(p(c)²)
  
  Information Gain is computed as:
  
    Gain = Entropy(S) - (|S_left|/|S|) * Entropy(S_left) + (|S_right|/|S|) * Entropy(S_right)
  
  This algo uses greedy recursive partitioning to determine the best local splits at each node.
