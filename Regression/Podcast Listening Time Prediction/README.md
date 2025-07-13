# Overview

This notebook contains a complete machine learning pipeline designed to predict podcast listening time, using tabular user and content metadata. Built around a regression task from the Kaggle Playground Series (Season 5, Episode 4) competition, this notebook explores various modeling strategies to forecast how long a user might spend listening to a podcast episode. The goal is to accurately estimate the listening duration based on features like genre, release time, user behavior, and ad placement.

# Libraries used


| Library                      | Purpose                                                                 |
|-----------------------------|-------------------------------------------------------------------------|
| `numpy`                     | Numerical operations and array handling                                 |
| `pandas`                    | Data loading, manipulation, and CSV I/O                                 |
| `warnings`                  | Suppresses warnings for cleaner output                                  |
| `matplotlib.pyplot`         | Visualization of data and model evaluation metrics                      |
| `seaborn`                   | Enhanced statistical plotting and visualizations                        |
| `train_test_split` (from `sklearn.model_selection`) | Splits dataset into training and testing subsets              |
| `OneHotEncoder` (from `sklearn.preprocessing`)      | Encodes categorical variables using one-hot encoding           |
| `SimpleImputer` (from `sklearn.impute`)             | Handles missing data by imputing values                         |
| `mean_squared_error` (from `sklearn.metrics`)       | Evaluation metric for regression tasks                          |
| `LinearRegression` / `Ridge` (from `sklearn.linear_model`) | Basic and regularized linear regression models         |
| `RandomForestRegressor` (from `sklearn.ensemble`)   | Ensemble model using multiple decision trees                    |
| `GradientBoostingRegressor` (from `sklearn.ensemble`)| Boosted trees focusing on reducing previous errors              |
| `AdaBoostRegressor` (from `sklearn.ensemble`)       | Weighted boosting method using weak learners                    |
| `StackingRegressor` (from `sklearn.ensemble`)       | Combines multiple models using a meta-regressor                 |
| `XGBRegressor` (from `xgboost`)                     | High-performance gradient boosting framework                    |
| `LGBMRegressor` (from `lightgbm`)                   | Fast and scalable gradient boosting model                       |
| `StandardScaler` (from `sklearn.preprocessing`)     | Normalizes features to zero mean and unit variance              |


# EDA

The dataset contains structured information about podcasts and user interaction, including:

Features: id, Podcast_Name, Episode_Title, Episode_Lenght_minutes, Genre, Host_Popularity_Percentage, Publication_Day, Publication_Time, Guest_Popularity_percentage, Number_of_Ads, Episode_Sentiment

Target: Listening_Time, a continuous value indicating how long a user listened to an episode (in minutes)

Initial exploration includes:

1. Data loading using pandas, value counts for categorical features like Genre, and statistical summaries for continuous features such as Number_of_Ads.

2. Number of Ads: Display a Boxplot, fill the missing values using mode, replace values greater than 3 with the mode, display the new Boxplot.

3. Take the null count of all feature columns.

4. Guest_Popularity_percentage: Display a statistiacl summary, take a count of values greater than 100, fill missing values using median. 

5. Compute the listening efficiency using Listening_Time_minutes / Episode_Length_minutes, plot a graph showing **Average Listening Efficiency per Genre**.

6. Genre: Plot a Genre Distribution graph

7. Episode length: Get a statistical summary and the Boxplot, Fill the missing values with median, Replace outliers(value > 120) with median.

8. Convert Episode_Length_minutes and Guest_Popularity_percentage into numeric.

9. Add a column named **Completion_Ratio**. (Listening_Time_minutes / Episode_Length_minutes)

10. Plot:
        
        Number of episodes per genre.
        
        Boxplot of completion ratio by genre.
        
        Average Listening Efficiency by Episode Sentiment.
        
        pivot_table of Listening Efficiency: Genre vs. Day of the week.
        
        barplot of Average Listening Efficiency by Day.
        
        pivot table of Listening Efficiency: Genre vs. Sentiment.

11. One-hot encode the categorical features:

        'Genre', 'Publication_Day', 'Publication_Time', and 'Episode_Sentiment.

12. Standardize the numerical columns:

        'Episode_Length_minutes','Host_Popularity_percentage','Guest_Popularity_percentage','Number_of_Ads','Listening_Efficiency','Completion_Ratio'

# Models used


| Model                     | Key Parameters                                                                 | RMSE     |
|---------------------------|----------------------------------------------------------------------------------|----------|
| **Random Forest Regressor** | `n_estimators=100`, `max_depth=None`, `random_state=42`                      | **0.1244** |
| **Gradient Boosting Regressor** | `n_estimators=100`, `learning_rate=0.1`, `max_depth=5`, `random_state=42`   | 0.3808   |
| **XGBoost Regressor**     | `n_estimators=100`, `learning_rate=0.1`, `max_depth=5`, `random_state=42`      | 0.6156   |
| **LightGBM Regressor**    | `n_estimators=100`, `learning_rate=0.1`, `max_depth=5`, `random_state=42`      | 0.6318   |

Random Forest Regressor gave the best results in terms of prediction accuracy (lowest RMSE).

Boosting models like Gradient Boosting, XGBoost, and LightGBM underperformed relative to Random Forest in this context, possibly due to the need for deeper hyperparameter tuning or limitations in the dataset size/complexity.

# Reasoning behind parameters

**1.Random Forest (max_depth=None)** (Why no depth limit?)

Random Forest is an ensemble of many deep, fully-grown trees, each trained on a random subset of data and features.

The goal is to let each tree explore the data thoroughly, even if it overfits its bootstrap sample. The ensemble effect — averaging across diverse trees — mitigates overfitting.

Hence, max_depth=None is often safe, and even beneficial, because variance is controlled through bagging and randomness.

**2.Boosting Models(max_depth=5)** (Why limit tree depth here?)

Boosting models like Gradient Boosting, XGBoost, and LightGBM build trees sequentially, where each tree tries to correct the errors of the previous one.

Deep trees in boosting can lead to severe overfitting, especially when combined with many boosting rounds.

Setting max_depth=5 is a regularization technique, limiting the complexity of each weak learner so that the ensemble learns more general patterns gradually.

Smaller depth ensures better generalization by preventing each boosting step from fitting noise or overly specific patterns.

**Conclusion**

The difference in max_depth settings reflects the different learning philosophies:

        1.Random Forest relies on deep, uncorrelated trees and uses bagging to control overfitting.
        
        2.Boosting models use many shallow, sequential trees and rely on iterative refinement. Shallow trees with depth=5 ensure they remain weak learners, which           is critical for boosting to work effectively.
        
# Feature importance

A bar chart named  **top 20 most important features** in predicting podcast listening time according to the XGBoost model was plotted. Feature importance here reflects how frequently and effectively a feature is used by the model's decision trees to split data and reduce prediction error. The most dominant feature was `Episode_Length_minutes`, which makes intuitive sense—longer episodes provide more opportunity for longer listening times. `Completion_Ratio` and `Listening_Efficiency` follow as the next most informative features, both directly tied to user engagement behaviors. Other features, like `Genre_Comedy`, `Publication_Day_Thursday`, and sentiment-related indicators (e.g., `Episode_Sentiment_Neutral`) contribute smaller, but still meaningful, predictive power. This plot helps highlight which variables the model prioritizes, guiding future efforts in feature engineering and data collection.


#  Stacking Regressor Configuration

| Model Name | Description                        | Key Parameters                                                                 |
|------------|------------------------------------|--------------------------------------------------------------------------------|
| `ridge`    | Ridge Regression                   | `alpha=1.0`                                                                    |
| `gbr`      | Gradient Boosting Regressor        | `n_estimators=100`, `learning_rate=0.1`, `max_depth=5`, `random_state=42`     |
| `xgb`      | XGBoost Regressor                  | `n_estimators=100`, `learning_rate=0.1`, `max_depth=5`, `random_state=42`     |
| `lgbm`     | LightGBM Regressor                 | `n_estimators=100`, `learning_rate=0.1`, `max_depth=5`, `random_state=42`     |
| **Meta-Model** | Linear Regression              | Used as the final estimator in the stacking ensemble                          |

Final Stacking Regressor Performance:

- **Validation RMSE**: `0.3536`

 # Testing

**1.Handling Missing & Invalid values:**

Episode_Length_minutes: Filled missing values with the median from the training set, Capped values greater than 120 minutes with the same median, Replaced zero values with NaN (to treat them as missing).

Number_of_Ads: Filled missing values using the mode of the training data, Replaced ad counts greater than 3 with the mode (assuming outliers).

Guest_Popularity_percentage: Missing values filled using the mean from the training set.

**2.Feature Scaling:**

Used StandardScaler to standardize the following numerical columns:Episode_Length_minutes, Host_Popularity_percentage, Guest_Popularity_percentage, Number_of_Ads

Fitted the scaler on training data and transformed the test set accordingly.

**3.One-Hot Encoding for Categorical Variables:**

Applied one-hot encoding on the following categorical columns:Genre, Publication_Day, Publication_Time, Episode_Sentiment

Used pd.get_dummies() to create binary columns.

**4.Column Alignment for Model Compatibility:**

Identified missing columns in the test set (present in X_train but not in test_df_encoded).

Added those columns with default value 0 to maintain shape consistency.

Reordered test_df_encoded columns to match the training feature set.

**Making Predictions:**

Used the trained **Random Forest Regressor** (rf_model) to predict podcast listening time on the test data.

Stored the predictions in a new column: Predicted_Listening_Time_minutes.

Created a submission file containing:id, Predicted_Listening_Time_minutes

Saved the output as submission.csv.

# Reasoning for final model selection 

The **Random Forest Regressor** was used in this project for several well-grounded reasons based on its strengths and how they align with the dataset and prediction task:



1. **Handles Nonlinear Relationships Well** :Podcast listening behavior is influenced by a complex mix of features like episode length, genre, day of publication, and ad count. These relationships are unlikely to be strictly linear. Random Forest, being an ensemble of decision trees, naturally captures such nonlinear interactions without requiring extensive feature engineering.

2. **Robust to Outliers and Noise**: Unlike linear models, which are sensitive to extreme values, Random Forest is relatively robust to outliers. This is particularly useful in scenarios like this one, where user listening patterns can vary widely and include occasional extreme durations.

3. **Performs Well with Mixed Feature Types**: The dataset includes both **numerical** features (like `Host_Popularity_percentage`) and **categorical** ones (like `Genre` and `Publication_Day` after encoding). Random Forest is effective at learning from this mix without the need for specialized transformations or assumptions.

4. **Minimal Tuning Required**: Compared to boosting models like XGBoost or LightGBM, Random Forest often delivers solid baseline results with fewer hyperparameter adjustments. This makes it a good initial choice for benchmarking model performance.

5. **Feature Importance Insight**: Random Forest provides clear feature importance scores, helping identify which features most influence predictions. This was leveraged in the notebook to visualize the top predictors of podcast listening time.

6. **Empirical Performance**: Most importantly, in this particular project, the Random Forest Regressor achieved the **lowest RMSE (0.1244)** on the validation set—outperforming other models like Gradient Boosting, XGBoost, and LightGBM. This strong performance validated its selection as the preferred model for generating final predictions.

Overall, Random Forest was chosen because it offered a strong balance of predictive power, interpretability, and reliability for the regression task at hand.

# Random Forest outperformed the Stacked Regressor because:

**It was already a strong standalone learner for this dataset.Random Forest is a strong learner that can handle both linear and nonlinear relationships, interactions between variables, and feature importance automatically. If the patterns in your dataset are:**

        1.Not extremely complex
        
        2.Not very high-dimensional
        
        3.Well-handled by decision trees

then Random Forest alone may already produce near-optimal predictions. In such cases, combining it with weaker or redundant models in a stacked ensemble may add complexity without improving generalization.

**The stacked model introduced unnecessary complexity.**

Stacking Complexity Can Backfire on Small or Clean DatasetsStacking introduces another layer of modeling — a meta-model that learns from the outputs of base models. This adds flexibility, but also:

        1.Increases the risk of overfitting, especially if the dataset is small or the models aren't diverse.
        
        2.Requires careful calibration of meta-model inputs and cross-validation.
        If the data isn’t large or complex enough, stacking may just reproduce the predictions of the strongest base model — and worse, may slightly degrade them.


The base models were not sufficiently diverse.

The meta-model was likely too simple to add meaningful improvement.

In short, sometimes simpler is better, and Random Forest gave the best balance of power, stability, and generalization for this task.















  










  



