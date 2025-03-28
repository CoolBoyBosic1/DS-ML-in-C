**NaiveBayes.cpp**  
- **model**: NaiveBayes

**Description**

  Implemented Naive Bayes for discrete and continuous features.

  NaiveBayes is a model based on Bayes Theorem

    P(A|B) = (P(B|A) * P(A)) / P(B)
  
  Where:
  
  **P(A)** is the apriori probability (computed as the frequency of the class in the training set).
  
  **P(B|A)** is the conditional probability of the features given class.
  
    For discrete features, P(B|A) is estimated as the frequency of each feature value in the class.
    For continuous features, P(B|A) is estimated using Kernel Density Estimation to compute the probability density.
    
  **P(B)** is constant across classes and can be ignored for classification.
  
  The classifier computes the posterior probability for each class and selects the class with the highest probability.
