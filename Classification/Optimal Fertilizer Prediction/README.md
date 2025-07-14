# Overview

This project focuses on the development and evaluation of machine learning models for predicting the most appropriate type of fertilizer to be applied, based on soil conditions and crop requirements. The challenge is derived from Kaggle’s Playground Series (Season 5, Episode 6), supplemented with original real-world data to enhance generalization and address class imbalance.

The goal is to build a model that accurately classifies the optimal fertilizer type given a set of agronomic features such as soil nutrients (N, P, K), temperature, humidity, pH, and moisture content. Correct fertilizer recommendations can greatly improve crop yield while minimizing environmental impact.

The task is multi-class classification, predicting one of the seven fertilizer categories. The project incorporates both synthetic competition data and real-world original data to improve robustness and model generalization.

# Dataset

1. Train/Test Data: Provided via the Kaggle competition, including feature columns like temperature, soil nutrients, moisture, and crop type.

2. Original Fertilizer Dataset: A supplemental dataset used to enrich the training set and address class imbalance issues, particularly for underrepresented fertilizers like DAP and Urea.

# Libraries used

| **Library**             | **Purpose / Description**                                                                                   |
|-------------------------|-------------------------------------------------------------------------------------------------------------|
| `numpy`                 | Fundamental package for numerical computing in Python. Used for array manipulation and numerical operations.|
| `pandas`                | Essential for data manipulation and analysis, especially DataFrame operations.                              |
| `LabelEncoder`          | Encodes categorical labels into integers. Helps prepare labels for classification models.                   |
| `MinMaxScaler`          | Scales numerical features to a given range (default 0 to 1). Helps normalize input data.                    |
| `StandardScaler`        | Standardizes features by removing the mean and scaling to unit variance.                                    |
| `KFold`, `StratifiedKFold` | Utilities for performing cross-validation, ensuring robust performance estimation.                        |
| `train_test_split`      | Splits dataset into training and testing sets.                                                              |
| `accuracy_score`, `precision_score`, `recall_score`, `f1_score` | Metrics for evaluating classification model performance.                                             |
| `classification_report`| Summarizes key metrics (precision, recall, F1-score) for each class.                                         |
| `confusion_matrix`, `ConfusionMatrixDisplay` | Computes and visualizes confusion matrix to show correct/incorrect predictions.                     |
| `XGBClassifier`         | High-performance gradient boosting classifier from the `xgboost` library.                                   |
| `CatBoostClassifier`    | Gradient boosting classifier optimized for categorical features and fast training from `catboost`.           |
| `seaborn`               | Statistical data visualization library based on matplotlib; useful for heatmaps, barplots, etc.             |
| `matplotlib.pyplot`     | Core plotting library for visualizing data (used to generate static plots and graphs).                      |
| `ipywidgets`            | Allows creation of interactive widgets in Jupyter notebooks (e.g., sliders, dropdowns).                     |
| `IPython.display`       | Provides tools for displaying images, widgets, and HTML in notebooks.                                       |
| `warnings`              | Built-in Python module used to suppress or manage warning messages.                                         |

# Data processing

Target label encoding using LabelEncoder.

Feature scaling via MinMaxScaler and StandardScaler depending on modeling needs.

Addressed class imbalance by augmenting the dataset (5x replication) and strategically oversampling classes with poor representation.

# Model

This project uses XGBoost (Extreme Gradient Boosting), a powerful and efficient machine learning algorithm, to predict the optimal fertilizer based on categorical agricultural features. The modeling pipeline includes robust evaluation through Stratified K-Fold Cross-Validation and focuses on performance metrics like macro F1-score and MAP@3 (Mean Average Precision at top 3).

**Training Strategy:** A StratifiedKFold with 5 splits was used to ensure each fold maintains the same class distribution as the original dataset. This is critical because the classes (fertilizer types) may be imbalanced.

For each fold:

1. The data was split into training and validation sets using the indices from StratifiedKFold.

2. To augment the training data, the original dataset (which likely represents high-quality or clean data) was concatenated with the current fold's training set.

3. The target column (Fertilizer Name) was separated and label-encoded, while all other features were explicitly converted to categorical types. This is important because XGBoost now supports native categorical handling when enable_categorical=True.

**Model Configuration**

The model is built using XGBClassifier with the following hyperparameters:

| **Parameter**        | **Value**          | **Description**                                                                     |
|----------------------|--------------------|-------------------------------------------------------------------------------------|
| `max_depth`          | 7                  | Controls the depth of each decision tree; deeper trees can capture more complexity. |
| `colsample_bytree`   | 0.4                | Fraction of features to consider per tree; helps with regularization.               |
| `subsample`          | 0.8                | Fraction of samples to use for training each tree; prevents overfitting.            |
| `n_estimators`       | 20000              | Maximum number of boosting rounds; early stopping is used to prevent overfitting.   |
| `learning_rate`      | 0.01               | A smaller learning rate ensures gradual updates, improving generalization.          |
| `gamma`              | 0.26               | Minimum loss reduction required for a split; controls model complexity.             |
| `max_delta_step`     | 4                  | Helps with logistic regression stability in imbalanced data.                        |
| `reg_alpha`          | 2.7                | L1 regularization term on weights.                                                  |
| `reg_lambda`         | 1.4                | L2 regularization term on weights.                                                  |
| `objective`          | `'multi:softprob'` | Multi-class classification using probability outputs.                               |
| `enable_categorical` | `True`             | Activates native support for categorical features.                                  |
| `tree_method`        | `'hist'`           | Efficient histogram-based training algorithm.                                       |
| `device`             | `'cuda'`           | Model is trained using GPU acceleration (if available).                             |

Training was performed using early stopping, with a patience of 100 rounds. This ensures that training halts once the validation performance stops improving, preventing overfitting and saving time.

**Evaluation Metrics**

**1.Macro F1-Score:** Measures the harmonic mean of precision and recall across all classes equally. Suitable for imbalanced class distributions.

**2.MAP@3 (Mean Average Precision at top 3):** Reflects the model’s ability to include the correct class within its top-3 predictions. This is particularly useful when partial correctness is valuable in recommendation scenarios.

**Loss curves**

1. The training loss decreases more steeply than the validation loss, which is expected as the model can overfit the training data.

2. The validation loss flattens out, indicating that further training provides minimal improvement and may risk overfitting.

3. The use of early stopping (e.g., early_stopping_rounds=100) would help terminate training around this plateau to avoid unnecessary computation and potential overfitting.

4. The model is learning effectively, with both losses declining. However, the gap between train and validation loss and the eventual flattening of validation loss signal the importance of controlling for overfitting—this curve validates that early stopping was a good practice.

 # Result analysis

**Confusion Matrix** :While the model exhibits decent discriminative ability, especially for certain fertilizers (like "14-35-14" and "17-17-17"), the relatively low diagonal scores and off-diagonal confusion suggest that the feature space is not highly separable across classes. Enhancing feature engineering (e.g., incorporating crop-specific or environmental features) or applying more advanced ensemble methods might further boost performance. Additionally, class-wise data balancing and domain-driven feature enrichment could help reduce misclassification.

**Classification Report**

An accuracy of 24.2% in a 7-class classification problem is only marginally better than random guessing (which would yield ~14.3%).

Indicates the model has learned some useful patterns but lacks strong discriminative power.

The support is relatively balanced across classes (~18k to ~23k per class), so poor performance is not due to class imbalance.

This points to limitations in feature quality, model complexity, or possibly noise in labels or input data.

While the model performs slightly better on certain classes (especially class 2), overall results are modest. With all metrics below 0.27, there's clear room for improvement through:

Enhanced feature engineering,

Incorporating domain-specific knowledge, or

Using more powerful ensemble or transformer-based architectures.

Better separation in the feature space or deeper models with regularization may also help push the performance upward.

**Feature Importance**:Since fertilizers are mostly defined by their composition, it is logical to see that soil nutrients play a big role in product selection. They need to complement each other.

CatBoost + Original Data: Phosphorus is the most important factor, followed by nitrogen. Potasssium is much lower in the ranking, just behind moisture and crop type.
CatBoost + 5x Original Data: Phosphorus remains on top, now followed by Moisture, while Nitrogen slips to third. Potasssium rose to fourth.

**Top-N Coverage**:The sharp increase from Top-1 to Top-3 (24.2% → 56.5%) suggests that although the model often fails to place the correct label at the very top, it does rank it among the more likely options.While the model's exact prediction accuracy is low, the Top-N Coverage shows promising ranking capabilities. This implies that the model has learned useful patterns, but it still struggles with precise final classification — likely due to feature overlap or subtle inter-class similarities. Improving input features, using ensembling or boosting with domain-specific features could enhance both Top-1 accuracy and general performance.

# Further improvements possible

Here are several **practical suggestions** to improve the model's performance for optimal fertilizer prediction:

1. **Feature Engineering Improvements**

* **Domain-specific ratios**: Derive meaningful features like N-P-K ratios, or ratios between soil type, crop, and moisture.
 
* **Interaction terms**: Create new features based on interactions (e.g., `soil_type * crop`, or `crop + temperature`).
  
* **Dimensionality reduction**: Use PCA or autoencoders to reduce noise and highlight informative signals.

2. **Data Preprocessing Enhancements**

* **Balance the dataset**: Use techniques like **SMOTE**, **class-weighting**, or **undersampling** to address class imbalance (as seen in the confusion matrix).
 
* **Outlier detection**: Use clustering or isolation forests to remove or smooth unusual input records.
 
* **Categorical encoding**: Try **target encoding** or **ordinal encoding** instead of treating all variables as `category`.

 3. **Modeling Strategies**

* **Ensemble learning**: Combine XGBoost with **LightGBM**, **CatBoost**, and even **neural networks** via stacking or voting to capture diverse patterns.
  
* **Tune hyperparameters**: Use **Optuna**, **GridSearchCV**, or **Bayesian optimization** to find the best model configuration.
 
* **Label smoothing**: For probabilistic models, slightly adjusting one-hot labels can improve generalization.

4. **Loss Function and Objective Tweaks**

* **Custom loss function**: Introduce a loss function that penalizes high-ranking wrong predictions more than low-ranking ones to improve Top-N ranking.
 
* **Class-weighted loss**: Apply weights to loss based on class frequencies to counteract imbalance.

 5. **Cross-Validation & Evaluation**

* Use **stratified cross-validation** with more folds (e.g., 10 instead of 5) to get more stable estimates.
  
* Consider **nested CV** to reduce overfitting during model selection.
 
* Optimize for **MAP\@3** or **Top-K accuracy** directly if it's more important than Top-1 accuracy in your application.

 6. **Deep Learning Alternatives**

* Experiment with **TabNet**, **TabTransformer**, or **Wide & Deep Networks** that can handle tabular data effectively while modeling interactions.
  
* Try **autoML frameworks** like AutoGluon, H2O.ai, or TPOT which explore a large space of models and preprocessing steps automatically.

 7. **Incorporate External Data**

* Add **weather data**, **seasonal patterns**, or **geographical features** if available.
  
* Use **crop yield statistics** or **soil health indexes** as supplementary features.

By integrating these strategies, especially a combination of **feature improvement**, **balanced training**, and **advanced modeling**, we can significantly increase Top-1 accuracy while retaining or improving MAP\@3 and Top-N coverage.


















