# Overview

This project provides a complete end-to-end pipeline for predicting house sale prices using machine learning. The dataset comes from the well-known Kaggle competition: House Prices – Advanced Regression Techniques. The objective is to predict the final selling price of a house based on various features such as lot size, number of rooms, quality indicators, and neighborhood characteristics. This notebook specifically uses TensorFlow Decision Forests (TF-DF) — a framework that brings classical decision tree models like Random Forest and Gradient Boosted Trees into the TensorFlow ecosystem.

# Libraries used

| Library                      | Purpose                                                                 |
|-----------------------------|-------------------------------------------------------------------------|
| `pandas`                    | Data loading, cleaning, and manipulation                                |
| `numpy`                     | Numerical operations and data splitting                                 |
| `seaborn`                   | Visualization of distributions (e.g., SalePrice)                        |
| `matplotlib.pyplot`         | Plotting histograms and feature importance                              |
| `tensorflow`                | Backend framework for model building                                    |
| `tensorflow_decision_forests` | Building and training the Random Forest regression model               |
| `IPython.display`           | Displaying visual outputs inside the notebook                           |
| `random`                    | Used in the manual dataset splitting function                           |

# Dataset loading and preprocessing

The training dataset is loaded from a CSV file containing features and the target variable SalePrice.

Basic inspection (.head(), .info()) is used to understand the structure and shape of the data.

The Id column, which is not useful for prediction, is dropped.

Distribution of the SalePrice column is visualized using Seaborn to understand the target variable’s range and skewness.

The dataset is split into numerical and categorical features for visualization.

Histograms of numerical features help identify value ranges and any potential outliers.

Rather than using built-in functions like train_test_split, the notebook defines a custom function using NumPy to randomly split the dataset into training and validation sets. This manual control ensures reproducibility and provides flexibility for more advanced split logic if needed later. The dataset is split into a 70:30 ratio between training and validation, which is typical for structured prediction tasks.

# EDA

The dataset undergoes a thorough EDA phase to understand its distribution and structure. First, the SalePrice column is analyzed using .describe() to check basic statistics like mean, min, max, and percentiles. A distribution plot is then used to visualize the skew and spread of sale prices. This helps understand if the data is normally distributed or skewed, which may influence model performance.

Data types are listed and separated into numerical and categorical features. Numerical features are selected using select_dtypes, and their histograms are plotted to identify outliers and data ranges. These visualizations help assess whether any transformations (e.g., normalization or log-scaling) are necessary. In this case, the decision forest model's robustness to scaling makes such preprocessing optional.

# TensorFlow Decision Forests

TF-DF is chosen as the modeling framework due to its non-neural, tree-based architecture, which excels in tabular data settings. Unlike deep learning models that require heavy preprocessing and careful feature scaling, decision forests are more robust to feature distribution and missing values.

The dataset is converted from pandas DataFrames to TensorFlow datasets using pd_dataframe_to_tf_dataset, which is a necessary step since TF-DF expects tf.data.Dataset input. The model is defined using the RandomForestModel class with the task set to REGRESSION, indicating a continuous output.

The model is compiled with Mean Squared Error (mse) as the metric, although this is optional because TF-DF automatically tracks key evaluation metrics internally.

# Model Training and Evaluation

Model training is performed using .fit() on the training dataset. Since decision forest models do not use backpropagation, training is fast and does not require epochs or batch sizes. After training, the internal structure of the decision forest is visualized using plot_model_in_colab, which displays the splits and feature thresholds used in a sample decision tree. This is a major advantage of TF-DF — it provides interpretability, which is often lacking in deep learning models.

To evaluate performance during training, the model logs the Root Mean Squared Error (RMSE) across the number of trees. A line plot is created to visualize how RMSE improves as more trees are added to the ensemble. This helps assess whether the model is underfitting or overfitting and aids in tuning tree-related hyperparameters.

The final model is evaluated on the validation dataset using .evaluate(). Key metrics such as RMSE and loss are printed, offering a direct measure of how well the model is generalizing.

# Model Interpretability

One of the standout features of TensorFlow Decision Forests is its built-in support for feature importance metrics. The notebook uses make_inspector() to access variable importance scores like:

NUM_AS_ROOT: how often a feature is used at the root of a tree,

SUM_SCORE: the cumulative gain across all splits involving the feature.

These scores are used to determine which features have the most influence on the model’s predictions. A bar plot is created to display the top features by their root usage, giving insight into which factors most significantly affect house prices. This is not just useful for model understanding, but also for feature selection in future experiments.














