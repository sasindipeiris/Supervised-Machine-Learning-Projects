# Predicting-Calorie-Expenditure
This project presents a comprehensive machine learning pipeline to predict calorie expenditure using a variety of biometric and physical activity-related features. The objective is to build and compare multiple models capable of accurately estimating the calories burned by an individual during physical activity. This type of modeling is crucial in fitness apps, personalized healthcare systems, and wearable technologies.

The task involves using a dataset containing various features (e.g., height, weight, BMI, and activity metrics) to predict the total calories burned. This is treated as a supervised regression problem. Multiple models, both linear and non-linear, are trained and evaluated to determine the best performing approach.

# Dataset
The dataset consists of individual samples with features such as age, weight, height, duration of activity, heart rate, and more. The target variable is the number of calories burned — making this a supervised regression problem. The model’s task is to learn the complex mapping between physiological inputs and energy expenditure.

# Libraries Used
The notebook makes use of:

NumPy and Pandas for efficient data manipulation.

Scikit-learn for preprocessing, modeling, and validation.

Advanced gradient boosting libraries — XGBoost, LightGBM, and CatBoost — for high-performing non-linear modeling.

Matplotlib for visualization

# Metric used
A custom evaluation metric, Root Mean Squared Log Error (RMSLE), is used.

The formula for **Root Mean Squared Logarithmic Error (RMSLE)** is:

$$
\text{RMSLE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} \left( \log(1 + \hat{y}_i) - \log(1 + y_i) \right)^2 }
$$

### Where:

* $n$ is the number of samples.
* $\hat{y}_i$ is the predicted value for the $i^{th}$ sample.
* $y_i$ is the true (actual) value for the $i^{th}$ sample.
* $\log(1 + x)$ is used (i.e., `log1p(x)`) to avoid issues when $x = 0$, since $\log(0)$ is undefined.

This formula ensures that predictions and targets are compared on a logarithmic scale, emphasizing **relative errors** and handling **skewed distributions** more effectively than RMSE.

RMSLE is preferred over standard RMSE because:

1.**It emphasizes relative errors, which is important for skewed targets like calories.**

One of the key benefits of RMSLE is that it focuses on the relative difference between the predicted and actual values, rather than just the absolute difference. This means it cares more about how much you’re off in percentage terms—for example, predicting 200 when the actual is 100 is worse than predicting 120 when the actual is 60, even though both errors are 100 units. This is especially useful when the target values are skewed or cover a large range, which is common in calorie data.
In skewed data, predicting a high-calorie burn slightly wrong (e.g., predicting 900 instead of 1,000) is not as bad as completely missing a small value (e.g., predicting 100 instead of 20). RMSLE captures this nuance by penalizing percentage differences, not raw differences.
In a skewed dataset, a few very large values could dominate a metric like RMSE. RMSLE reduces their dominance by applying a log transformation, which helps create a fairer evaluation across the entire range.

2.**It penalizes underestimates more harshly than overestimation, which is safer in health applications.**

In health applications, underestimating calories burned can be risky because it might lead someone to eat too less or overtrain based on wrong feedback. 

# Feature Engineering and data preprocessing

1.**Add Domain based feature(Calories_Burned)** for both train_df and test_df.It created a new column Calories_Burned using sex-specific formulas derived from heart rate, age, weight, and duration.The formula differs for males and females and is scaled by dividing by 4.184 to convert from kilojoules to kilocalories.

2.**Create Duration Bins**-The Duration column was discretized into bins of size 5 using pd.cut().New column Duration_class was created for both train_df and test_df.

3.**Create Age Bins**-Similarly, the Age column was binned into intervals of 5 years.The resulting age_class column was added to both train_df and test_df.

4.**Map sex to numerical format**-The categorical Sex column ('male'/'female') was mapped to numeric values: male → 1, female → 0.The resulting column was explicitly converted to type float32.

5.**One-Hot Encode Duration_class and age_class**-A OneHotEncoder was used to convert the two categorical columns into binary columns for each bin.The original Duration_class and age_class columns were dropped and replaced with the one-hot encoded columns.

6.**Split features and labels for training**-Columns id and Calories were dropped from train_df to form the input matrix X.The target column Calories was stored as y.In test_df, only the id column was dropped to create X_test.

7.**Group based target encoding**-The median Calories was calculated for each unique combination of Sex, age_class, and Duration_class.The aggregated result was merged back into both train_df and test_df as a new feature: Calories_encoded.

# Validation Stategy
A Repeated K-Fold Cross-Validation strategy is used:

5 splits × 3 repeats = 15 training/validation cycles.

This robust strategy reduces variance in performance estimates and ensures better generalization.

A fixed random_state is used for reproducibility.

# Models used initially

1.**CatBoost Regressor**
A high-performance gradient boosting framework:

Handles categorical and numerical features natively

Uses ordered boosting to reduce overfitting

Requires minimal preprocessing

2.**LightGBM Regressor**
An efficient, fast boosting method using histogram-based learning:

Great for large datasets

GPU acceleration

Tunable with many hyperparameters

3.**XGBoost Regressor**
Another robust boosting library:

Incorporates regularization

Effective for tabular data

# Usage of stacking
In this notebook, stacking is used as an advanced ensemble learning technique to improve the accuracy of calorie expenditure predictions by intelligently combining the strengths of multiple regression models.Specifically, tree base models—CatBoost Regressor,LightGBM Regressor,XGBoost Regressor—are first trained independently on the same training data. Instead of directly using their predictions as final outputs, each model’s predictions on the training and test sets are collected and treated as new input features. These model-generated predictions are then assembled into new training and test datasets, effectively transforming the problem into one where the “meta-features” represent the outputs of the base models. 

# Final Prediction - Using Multiple Meta Models

1.**Ridge Regression** -A linear model with L2 regularization that helps avoid overfitting and handles multicollinearity in features.

2.**Linear Regression** -Standard Linear Regression provides a basic performance benchmark.

3.**Bayseian Ridge Regression**-Bayesian Ridge Regression introduces uncertainty estimation in weights and offers more stability for small datasets.

After performing model stacking to generate a new set of features based on the predictions of three base regressors, this final block of code takes the ensemble approach one step further by introducing multiple **meta-models** and averaging their outputs to form the final prediction. Instead of relying on just a single model to combine the stacked features, three different regression algorithms are trained on the transformed training set (`stacked_train`) using the **log-transformed target variable** (`y_log`), which applies `log1p()` to stabilize variance and reduce the impact of large calorie values. The three meta-models used are: **Ridge Regression** with cross-validated regularization strength, **Linear Regression** for an unregularized baseline, and **Bayesian Ridge Regression**, which incorporates probabilistic regularization through Bayesian priors and performs iterative fitting with hyperparameter tuning. Each model produces its own predictions on the stacked test data (`stacked_test`), and these predictions are then averaged to form a single, robust estimate for each sample. Finally, since the models were trained on log-transformed data, the predictions are reverted back to their original scale using `expm1()`, ensuring accurate calorie predictions. This ensemble of meta-models helps reduce individual model bias, improves generalization, and enhances overall performance by leveraging the strengths of each approach.

# Conclusion

This notebook presents a comprehensive pipeline for predicting calorie expenditure using a combination of feature engineering, ensemble learning, and meta-modeling. 

1.The process begins with thoughtful feature extraction, where domain-specific knowledge is used to derive a new `Calories_Burned` feature based on sex-specific physiological formulas incorporating heart rate, age, weight, and duration. Additional preprocessing steps include converting continuous variables like `Age` and `Duration` into categorical bins and applying one-hot encoding to these for better model interpretability. A group-based median target encoding (`Calories_encoded`) is also introduced, which captures high-level trends across demographic segments. 

2.The modeling phase involves training three base regressors—Gradient Boosting, Random Forest, and XGBoost—whose predictions on the training and test sets are used to form a stacked dataset. Instead of relying on a single model to interpret these stacked features, three different meta-models—Ridge Regression, Linear Regression, and Bayesian Ridge Regression—are trained on the log-transformed target to mitigate skewness and improve numerical stability. Their outputs are then averaged to form the final prediction, which is converted back to the original scale using `expm1`. RMSLE (Root Mean Squared Logarithmic Error) is used as the evaluation metric due to its ability to emphasize relative error, handle skewed data effectively, and penalize underestimation more strongly—important for health-related predictions. 

Altogether, the notebook demonstrates a robust strategy that blends theoretical understanding, feature synthesis, and ensemble learning to yield accurate and reliable predictions of calorie expenditure.




































