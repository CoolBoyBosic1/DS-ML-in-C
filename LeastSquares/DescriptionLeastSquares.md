**LeastSquares**
  **model**: least squares
  **loss function**: sum of squared errors
  **optimization function**: analytical solution

**Description**
  Least Squares is a method that finds the optimal linear model by minimizing the sum of squared differences between the actual values and the predicted values.
  
  least squares is a model computed as
  
    y = b₀ + b₁ * x  (where b₀ is the intercept b₁ is the slope)
  
  sum of squared errors is a loss function computed as
    
    SSE = Σ (yᵢ - (b₀ + b₁ * xᵢ))²
  
  Optimization is done via analytical solution for the parameters is obtained by solving the normal equations:
  
    b₁ = [Σ (xᵢ - mean(x)) (yᵢ - mean(y))] / [Σ (xᵢ - mean(x))²]  
    b₀ = mean(y) - b₁ * mean(x)
  
  This method computes the optimal parameters directly without iterative optimization.
