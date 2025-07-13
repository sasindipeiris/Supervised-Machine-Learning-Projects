# Overview

This project addresses the task of forecasting daily sales for various products across multiple store locations. It was built using a comprehensive time series dataset provided by a Kaggle competition, which includes historical sales, promotions, holidays, oil prices, and store metadata. The aim is to build a model that can generalize well to unseen dates and predict unit sales per store-family combination with accuracy.

# Datasets

The notebook loads six main datasets:

1. train.csv: historical sales data

2. test.csv: data points for which predictions are required

3. stores.csv: metadata about each store (type, location, cluster)

4. oil.csv: daily oil prices, as a macroeconomic indicator

5. holidays_events.csv: holiday calendar including special days

6. transactions.csv: number of daily transactions per store

All dates were parsed properly for time series processing.

# Libraries used


| Library       | Purpose                                                                 |
|---------------|-------------------------------------------------------------------------|
| `numpy`       | Provides support for numerical operations and array structures          |
| `pandas`      | For loading, manipulating, and analyzing tabular data                   |
| `matplotlib`  | Core plotting library used for visualizations                           |
| `seaborn`     | Statistical data visualization based on matplotlib                      |
| `lightgbm`    | Gradient boosting framework for efficient, high-performance modeling    |

# EDA

The basic EDA steps include:

1. Check for missing values in each data frame.
 
2. Print the summary statistics of the training data, transaction data, 

3. Plot:

  Distribution of Sales
  Total sales aggregated by date
  Distribution of Onpromotion Counts
  Distribution of Transactions
  Total Sales by Store
  Sales Distribution by Product Family
  Average Sales by Product Family
  Oil Prices Over Time
  Total Sales on Holiday vs Non-Holiday Days
  Average Sales on Holiday vs Non-Holiday Days

# Feature Engineering

1. Combine train and test for consistent feature engineering. (Create a dummy sales column in test.)

2. Merge External Data:
   
  Stores Data
  Oil Data
  Transaction Data
  Holidays Data(Remove any existing "is_holiday" column before merging holidays to avoid duplicates)

3.Create lag features for sales within each store and product family group.

4.Create a 7-day rolling average of past sales.

5.Additional Feature: Ratio of onpromotion items to transaction.

# Data Preprocessing

1. Handle missing values : Fill missing transactions and oil prices if any.

2. Seperate train and test data.

3. Drop raws with missing Lag features in training data.

4. Convert 'store_nbr', 'family', 'city', 'state', 'type' to categorical.

5. Split into train and validation sets based on date.

# Model training

The LightGBM library was used to build a time series regression model.

**1. Evaluation Metric : RMSLE**

The model was trained using the Root Mean Squared Logarithmic Error (RMSLE), which is particularly suited for predicting quantities like sales:

  RMSLE penalizes under-predictions more than over-predictions.
  
  It is ideal for skewed data where large targets are common but the model should still treat relative errors fairly.
  
  To prevent invalid log values, all predictions are passed through np.maximum(0, preds) to ensure non-negativity.
  
  
  
**2. Hyperparameters used:**

  'objective': 'regression' specifies a regression task.
  
  'metric': 'None' disables LightGBM’s default metrics to rely solely on the custom rmsle.
  
  'boosting': 'gbdt' enables gradient boosting decision trees.
  
  'learning_rate': 0.05' controls the step size at each iteration.
  
  'num_leaves': 31' defines the complexity of each tree (a moderate default).
  
  'seed': 42' ensures reproducibility.
  
  'verbose': -1' silences console logs for cleaner output.

**3. Training Strategy with Early Stopping**

The model was trained for a maximum of 1000 boosting rounds.

Early stopping was enabled using lgb.early_stopping(100), which halts training if no improvement is seen in the validation RMSLE over 100 rounds.

A callback lgb.record_evaluation() stored metrics for both train and validation sets.

**4. Results and Performance**

The LightGBM model was trained with early stopping for up to 1000 boosting rounds and completed training at iteration 950, which is the point where it achieved the best validation score based on the custom RMSLE metric.


Final scores were:

  Training RMSLE: 0.7982
  
  Validation RMSLE: 0.5748

This gap suggests the model fits training data reasonably well while still generalizing effectively to the validation set, with no strong indication of overfitting.

**Training RMSLE (0.7982)**

This score reflects the model’s error when predicting on the training dataset — data the model has already seen. A reasonably low score here shows that the model has learned meaningful patterns in the training data. However, if this value were much lower than the validation score, it could suggest overfitting.

**Validation RMSLE (0.5748)**

This score measures how well the model performs on unseen data — i.e., the validation set. A lower value here indicates that the model's predictions are closely aligned with actual sales for new inputs.

**Gap Analysis and Generalization**

The difference between training and validation RMSLE (approximately 0.22) is relatively small, which is a positive indicator:

  The model did not overfit drastically to the training data.
  
  It generalizes well to unseen data, which is crucial for real-world forecasting tasks.
  
  The slight gap is expected in any supervised learning model and indicates a good trade-off between bias and variance.

Overall, the training and evaluation outcomes suggest that the model is both accurate and robust, making it a strong candidate for forecasting store sales in a time series setting. Further tuning (e.g., feature selection, deeper cross-validation, hyperparameter optimization) could still be explored to reduce the validation error even further.











  
  










