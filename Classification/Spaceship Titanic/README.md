# Overview

This project addresses the Spaceship Titanic challenge, a binary classification task where the goal is to predict whether a passenger was transported to an alternate dimension during a fictional interstellar disaster. The task uses structured tabular data and serves as an ideal playground for applying TensorFlow Decision Forests (TF-DF), a powerful tree-based modeling library within the TensorFlow ecosystem.

The fictional setting is inspired by the Titanic disaster — but with a space-age twist. Passengers aboard the Spaceship Titanic may have either been transported or not during the accident. The objective is to use features such as age, home planet, cabin, spending habits, and other metadata to predict the binary target:

Transported = True if the passenger was transported

Transported = False otherwise

# Dataset Overview

The dataset is provided as a CSV file and contains:

**1. Categorical features**: HomePlanet, CryoSleep, Cabin, Destination, VIP

**2. Numerical features**: Age, RoomService, FoodCourt, ShoppingMall, Spa, VRDeck

**3. String features**: Name, Cabin (decomposed later)

**4. Target**: Transported (boolean)

Missing values are present and handled during preprocessing.

# Libraries used

**tensorflow, tensorflow_decision_forests** – for model building and training

**pandas, numpy** – for data manipulation

**matplotlib, seaborn** – for visualization

# EDA

1.  Dataset is loaded using Pandas.

2. .head(), .info(), .describe() methods were used to understand the structure and content.

3. A bar plot visualizes class balance for the target variable (Transported).

4. Histograms were plotted for Age, FoodCourt, ShoppingMall, Spa, VRDeck.

# Data Preprocessing

1. Features with missing values are imputed.

2. PassengerId,Name is dropped from the dataframe.

3. **Label Encoding**: The target column `Transported` (originally boolean) is converted to integer type (`True` → 1, `False` → 0) for compatibility with machine learning models.

4. **Boolean Conversion**: The categorical boolean columns `VIP` and `CryoSleep` are also cast to integers (`True`/`False` → `1`/`0`) for numerical processing.

5. **Cabin Feature Engineering**: The `Cabin` column, which contains entries like `"F/123/S"`, is split into three new columns: `Deck`, `Cabin_num`, and `Side` based on the "/" delimiter.

6. **Column Cleanup**: After splitting, the original `Cabin` column is dropped from the dataset using `drop()`. A `try-except` block ensures that if the column has already been removed, the script continues without error.


# TensorFlow Decision Forests Models

TF-DF models are particularly well-suited for tabular data and require less preprocessing compared to neural networks.

| Model Class                                                                 | Description                                                                                   | Use Case                                               |
|------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|--------------------------------------------------------|
| `tensorflow_decision_forests.keras.RandomForestModel`                        | Implements a classic Random Forest ensemble using multiple decision trees.                    | Robust baseline for classification and regression.     |
| `tensorflow_decision_forests.keras.GradientBoostedTreesModel`               | Implements Gradient Boosted Decision Trees with iterative training to reduce prediction error.| Best for tabular data where high accuracy is desired.  |
| `tensorflow_decision_forests.keras.CartModel`                                | Implements a single Classification and Regression Tree (CART).                                | Lightweight, interpretable decision tree.              |
| `tensorflow_decision_forests.keras.DistributedGradientBoostedTreesModel`    | Distributed version of Gradient Boosted Trees for large datasets and multi-node training.     | Scalable GBT for large-scale training environments.    |

 
# Model Evaluation

These two sets of evaluation metrics compare the **performance of your trained model** versus a **default (baseline) model**. Here's a breakdown of what each one means:


#**Trained Model Performance**

These values are based on predictions made by your **trained decision forest model**:

* **Accuracy: 0.7932**
  → The model correctly predicted about **79.3%** of the test examples.

* **CI95\[W]\[0.785031 0.801152]**
  → This is the **95% confidence interval** for accuracy, meaning there's 95% certainty that the true accuracy falls between **78.5% and 80.1%**.

* **LogLoss: 0.57859**
  → Measures the uncertainty of the predicted probabilities. Lower is better. A value of 0.578 shows the model assigns high confidence to correct predictions.

* **ErrorRate: 0.2068**
  → About **20.7%** of predictions were incorrect (complement of accuracy).



**Default (Baseline) Model Performance**

These values are for a **dummy classifier** (baseline model) that does **not learn** from data — it might always predict the majority class.

* **Default Accuracy: 0.5024**
  → This model is correct only about **50.2%** of the time — no better than guessing.

* **Default LogLoss: 0.6931**
  → This is the **maximum entropy** for binary classification with 50/50 guessing. It indicates poor confidence and accuracy.

* **Default ErrorRate: 0.4976**
  → About **49.8%** of predictions are wrong — very close to random guessing.


The trained model significantly outperforms the default model in all metrics, especially accuracy and log loss. This confirms that it’s learning useful patterns and generalizing well beyond random guessing.

# Out of Bag Accuracy vs Number of Trees

When building a Random Forest, each decision tree is trained on a random bootstrap sample (i.e., a sample with replacement) from the training dataset. This means that, on average, about one-third of the original training data is left out from each tree's training data — these left-out samples are called Out-of-Bag samples.

**How it's used**:

1. Each tree can make predictions only for its OOB samples.

2. These predictions are aggregated across all trees that did not use a given sample during training.

3. This gives an unbiased estimate of model accuracy, known as OOB accuracy, without needing a separate test set.

**Why it's useful**:

1. Acts like built-in cross-validation.

2. Saves time and data, since no extra hold-out set is needed.

3. Helps detect overfitting early in training.

**Initial phase (0 to ~50 trees):** The accuracy increases sharply. This is expected because adding more trees at the start helps the model learn more complex patterns and generalize better.

**Middle phase (~50 to 200 trees):** The accuracy continues to improve but at a slower rate, eventually reaching a plateau. This indicates that most useful patterns have already been captured, and additional trees are offering diminishing returns.

**Later phase (200 to 300 trees):** The accuracy slightly fluctuates or stabilizes. This suggests that adding more trees beyond this point does not significantly improve performance, and might even introduce some noise.

The model achieves its best performance with around 200 trees, and adding more trees beyond that shows diminishing gains. This graph is useful for identifying the optimal tree count, helping to avoid unnecessary computation while maintaining strong performance.

# Feature Importance

TF-DF automatically provides ranking of features based on their contribution to splits in the tree.

This helps interpret which passenger characteristics were most predictive of transportation.

These represent different ways TensorFlow Decision Forests quantifies **feature importance**:

* **SUM\_SCORE**: Total contribution of a feature to the model's split gains across all trees (higher means more predictive power).
  
* **NUM\_AS\_ROOT**: Number of times a feature appears as the root (first) split in trees, indicating strong early influence.
  
* **INV\_MEAN\_MIN\_DEPTH**: Inverse of the average depth at which a feature appears — features used closer to the root are more important.
  
* **NUM\_NODES**: Number of tree nodes where the feature is used for splitting, reflecting how frequently it's used across the forest.

# Conclusion

This notebook presents a complete machine learning pipeline for solving the Spaceship Titanic classification problem using **TensorFlow Decision Forests (TF-DF)**. The dataset was thoroughly preprocessed by handling missing values, converting categorical columns to numerical form, and engineering features like deck and side from the `Cabin` attribute. The model training was done using a `RandomForest`, and performance was evaluated using metrics such as **accuracy**, **log loss**, and **error rate**, with comparisons made against a default baseline model. Notably, the trained model achieved an accuracy of over **79%**, significantly outperforming the baseline. The notebook also explored **feature importance** using metrics like `SUM_SCORE` and `NUM_AS_ROOT`, and visualized how increasing the number of trees impacts out-of-bag (OOB) accuracy. The results demonstrate that decision forests are highly effective for structured data, offering strong performance with minimal preprocessing and intuitive interpretability.













