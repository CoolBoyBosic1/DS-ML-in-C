**MLP.cpp**

   **Model**: Multilayer Perceptron
   
   **Loss Function**: Mean Squared Error, Mean Absolute Error, Root Mean Squared Error
   
   **Activation Functions**: ReLU, Sigmoid, Tanh
   
   **Optimization Function**: Backpropagation via Gradient Descent



**Description**

Implemented MLP for multi-layer neural network processing.

The model is computed layer-wise:

  **Linear Transformation**
    z = w * x + b
    (where w - weight vector/matrix, x - input, and b - bias coef)

  **Activation Function**
    a = f(z)
    The activation function f can be:
      
      ReLU:  f(z) = max(0, z)
      Sigmoid: f(z) = 1 / (1 + exp(–z))
      Tanh:   f(z) = (exp(2z) – 1) / (exp(2z) + 1)

  The output from the final layer is compared to the target using the MSE:
    
    L = sum( (target – output)^2) / N (where N - num output neurons)

  To update the params (weights and biases), backpropagation:
    
    dL/dw = (dL/da) * (da/dz) * (dz/dw)
    dL/db = (dL/da) * (da/dz) * (dz/db)

  gradient descent update the parameters iteratively:
    
    w := w – l * (dL/dw)
    b := b – l * (dL/db) (where l is the learning rate)

