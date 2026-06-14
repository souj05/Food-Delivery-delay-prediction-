# Food-Delivery-delay-prediction-


## Problem Statement
Food delivery platforms need to identify orders that are likely to be delayed or not  so that they can take proactive measures and improve customer satisfaction.

## Machine Learning Pipeline

1. Data Loading
   - Loaded the food delivery dataset.

2. Data Cleaning
   - Handled missing values using median imputation.
   - Removed unnecessary identifier columns.

3. Feature Engineering
   - Created relevant features for modeling.
     

4. Data Leakage Detection & Removal
   - Identified leakage-related features.
   - Removed target-dependent variables and retrained models.

5. Data Preprocessing
   - Performed train-test split.
   - Prepared features for machine learning models.

6. Model Training
   - Logistic Regression
   - Random Forest
   - AdaBoost
   - XGBoost

7. Model Evaluation
   - Accuracy Score
   - Confusion Matrix
   - Classification Report

8. Model Validation
   - Cross-Validation

9. Hyperparameter Tuning
   - GridSearchCV for Random Forest optimization.

10. Feature Importance Analysis
   - Identified and visualized the most influential features affecting delivery delays.

11. Business Insights
   - Interpreted key factors contributing to delayed deliveries.
