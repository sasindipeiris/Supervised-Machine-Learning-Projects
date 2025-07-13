# Overview

This notebook presents a straightforward machine learning pipeline built to solve the Titanic - Machine Learning from Disaster classification challenge on Kaggle. The objective is to predict whether a passenger survived or not based on features like gender, passenger class, number of siblings/spouses aboard, and parents/children aboard.

# EDA

The notebook begins by calculating and printing survival rates for men and women separately:

A significantly higher percentage of women survived compared to men, highlighting gender as a strong predictor of survival.

This insight motivates the inclusion of "Sex" as a key feature in the model

# Model Building and Prediction

A Random Forest Classifier is used to train the model. Key steps include:

Selection of a feature subset: ["Pclass", "Sex", "SibSp", "Parch"]

Conversion of categorical variables to numerical form via pd.get_dummies()

Training the model using RandomForestClassifier with:

**n_estimators**=100 (number of trees)

**max_depth=5** (to control overfitting)

**random_state=1** (for reproducibility)

The model is then applied to the test set, and predictions are stored in a CSV submission file.

# Key Takeaways

1. Feature Simplicity: The model uses only a small subset of features but still captures important survival patterns (especially gender and class).

2. Model Choice: Random Forest provides a strong balance between accuracy and interpretability for this tabular dataset.

3. Effective Baseline: This notebook is a strong starting point that could be enhanced by feature engineering (e.g., using Age, Fare, or Cabin), handling missing data, or hyperparameter tuning.

