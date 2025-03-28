**SVM.cpp**  
   **Model**: Support Vector Machine  
   **Loss Function**: Hinge Loss  
   **Regularization**: L2 on weight  
   **Optimization Function**: Gradient Descent  



**Description**

  Implemented SVM for binary classification.
  
  SVM is a model which is computed as
  
    f(x) = w * x + b  (where w is a weight x is some sample and b coeficient)
    
  Hinge Loss is a loss function which is computed as
  
    L = max(0, 1 - y * f(x)) (where f(x) is a model in our case SVM and y is a test sample)
    
  L2 is a Regularization which is computed as
  
    Total loss = L +  λ * (w²) (where L is some loss function in our case Hinge loss,  λ - parameter and w - weight)
    
  Optimization is done via GredientDescent which is a classic derivative of total loss by weight and b which are updated on next iteration



