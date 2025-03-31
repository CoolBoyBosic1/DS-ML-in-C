**XGBoost.cpp**  

   **Model**: XGBoost  
  
   **Loss Function**: RMSE
    
   **Optimization Function**: gradient boosting (additive model with second-order Taylor approximation)


**Description**

Implemented a simplified version of XGBoost using decision trees as base learners

Algo:
  The base learner is a Decision Tree, built using greedy recursive partitioning based on impurity (entropy/djini).
  
  In each boosting iteration, the residuals between the current predictions and true labels are computed.
  
  A new decision tree is built on these residuals to correct the previous errors.
  
  The ensemble's prediction is updated additively:
    
    new_prediction = current_prediction + learning_rate * tree_output
     
