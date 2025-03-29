**LogisticRegression.cpp**

   **Model**: Logistic Regression
   **Loss Function**: Crossentopy olss 
   **Optimization Function**: RMSprop



**Description**

  Implemented LogisticRegression for binary classification

  Logistic Regression is a model which is computed as
  
    f(x) = 1/(1 + e^(x * slope + intercept)) (where x is a data point, slope is the weight, and intercept is the bias)

   Crossentopy is a loss which is computed as 
     
     L = -(y * log(p) + (1 - y) * log(1 - p))  (where p is the model's predicted probability(from 0 to 1) and y is the true label)

   RMSprop is an optimization function which is computed as

     θₜ₊₁ = θₜ - (l / √(E[g²]ₜ + ε)) · gₜ

    where:
      l is the learning rate, gₜ is the gradient at time t,
      E[g²]ₜ is the exponentially weighted moving average of squared gradients, computed as:
        E[g²]ₜ = d · E[g²]ₜ₋₁ + (1 - d) · gₜ²,
      ε is a small constant for numerical stability and d is the decay rate.
     
     
