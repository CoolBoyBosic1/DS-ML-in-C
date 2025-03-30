**RandomForest.cpp**  
   **Model**: Random Forest  
   **Components**: Decision Trees ensemble  
   **Bootstrap**: Sampling with replacement  
   **Voting**: Majority Vote  



**Description**
   Implemented Random Forest by combining multiple decision trees. 
   Each tree is built using a bootstrap sample of the training data. 
   For each tree, the best split is determined using the decision tree’s split function (based on entropy). 
   For prediction, each tree votes on the class label, and the final prediction is made by majority vote.

