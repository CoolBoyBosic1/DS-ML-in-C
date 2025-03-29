**LinearRegression**
   **model**: linear regression,
   **loss functions**: MAE, MSE, RMSE, 
   **optimization function**: gradient descent

**Description**

  Linear Regression is a model which is computed as

    f(x) = slope * x + intercept (where x is a datapoint, slope is a weight and intercept is a coeficient)

  MAE is a loss which is computed as 

    L = |y - f(x)| (where f(x) is a value which predicted model and y is a true value)

  MSE is a loss which is computed as

    L = (y - f(x)) ^ 2 (where f(x) is a value which predicted model and y is a true value)

  RMSE is a loss which is computed as MSE but taken squared root after computing

  Gradient descent is an optimization function which is a classic derivative of loss by weight and intercept 
  which are updated and ready for next iterations. for computing derivatives I used central difference method

    (f(x + h) - f(x - h)) / 2h
