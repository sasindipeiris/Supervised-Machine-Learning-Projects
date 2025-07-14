# Overview

This project explores the use of deep learning to differentiate between body-focused repetitive behaviors (BFRBs), such as hair pulling, and non-BFRB everyday gestures, like adjusting glasses. The challenge originates from a real-world competition that aims to leverage multi-sensor data—including movement (IMU), temperature (thermopile), and proximity (ToF) sensors—collected via a wrist-worn device to predict whether a given sequence corresponds to BFRB-like or non-BFRB-like activity.

The ultimate goal is to build a predictive model capable of identifying these behaviors with high accuracy. Successfully solving this task will lead to better-designed wearable devices for BFRB detection and significantly contribute to mental health monitoring technologies. Such systems can assist in early diagnosis, behavior tracking, and personalized interventions for a wide range of mental illnesses.

This repository presents a complete pipeline including data preprocessing, statistical feature extraction, sequence modeling using LSTMs, and model evaluation—all designed to improve the classification of behavioral patterns from raw sensor data.

# Libraries used



| Library / Module                         | Purpose                                                                 |
|------------------------------------------|-------------------------------------------------------------------------|
| `pandas`                                 | For loading and manipulating structured data in tabular form           |
| `numpy`                                  | For numerical computations and efficient array handling                |
| `sklearn.model_selection.train_test_split` | Splits the dataset into training and testing subsets                   |
| `sklearn.preprocessing.StandardScaler`   | Standardizes features by removing mean and scaling to unit variance    |
| `sklearn.preprocessing.LabelEncoder`     | Converts categorical labels into integer codes                         |
| `sklearn.metrics.classification_report`  | Generates detailed classification performance report                   |
| `sklearn.metrics.accuracy_score`         | Computes the accuracy classification metric                           |
| `scipy.stats.skew`                       | Measures asymmetry of the probability distribution of a dataset        |
| `scipy.stats.kurtosis`                   | Measures the tailedness of the distribution                            |
| `scipy.stats.iqr`                        | Computes the interquartile range, a measure of statistical dispersion  |
| `tensorflow`                             | Core deep learning framework used to build and train neural networks   |
| `tensorflow.keras.models.Sequential`     | A linear stack of layers for building sequential models                |
| `tensorflow.keras.layers.LSTM`           | Adds Long Short-Term Memory layers for sequence modeling               |
| `tensorflow.keras.layers.Dense`          | Fully connected layer used in deep learning models                     |
| `tensorflow.keras.layers.Dropout`        | Regularization technique to prevent overfitting                        |
| `tensorflow.keras.utils.to_categorical`  | Converts integer labels into one-hot encoded format                    |

# Dataset

The dataset includes three types of sensor channels:

IMU Sensors: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z

Thermopile Sensors: therm_1 to therm_4

ToF Sensors: tof_distance, tof_signal_strength

# Feature Engineering


* **Sliding Window Technique**:

  * Used to segment the time-series data.
    
  * Each window is 50 data points long.
    
  * Consecutive windows overlap by 25 points to maintain context and increase data coverage.

* **Window Validation**:

  * A window is considered valid if it:

    * Contains exactly 50 data points.
      
    * Has a consistent label (i.e., all rows in the window share the same behavior value).

* **Statistical Feature Extraction**:

  * For each sensor column in a valid window, 12 features are extracted:

    * Mean, Standard Deviation, Minimum, Maximum, Median, Variance
      
    * Root Mean Square (RMS), Signal Magnitude Area (SMA), Peak-to-Peak (PTP)
      
    * Interquartile Range (IQR), Skewness, Kurtosis
      
  * If a sensor is missing from the data, a zero-vector of 12 values is appended instead.

* **Data Structuring**:

  * Each window is transformed into a fixed-length feature vector.
    
  * These vectors are stored in `features`, while their corresponding behavior labels go into `labels`.
    
  * `window_indices` tracks the starting index of each valid window.

* **Final Output**:

  * Features (`X`) and labels (`y`) are converted to NumPy arrays for modeling.
    
  * Dimensions: `X.shape = (num_windows, total_features_per_window)`, `y.shape = (num_windows,)`
    
  * A total of `len(ALL_EXPECTED_SENSOR_COLUMNS) * 12` features are extracted per window.

# Preprocessing

1.**Encode Labels and Split Data**

* **Validation Before Encoding**:

  * Checks if the label array `y` exists and is a valid non-empty NumPy array.
    
  * If the check fails, it exits with an error message suggesting to verify the data loading and feature engineering steps.

* **Label Encoding**:

  * Uses `LabelEncoder()` from scikit-learn to convert string labels into integer format.
    
  * Applies `fit_transform()` to obtain encoded labels (`y_encoded`).
    
  * Computes the number of unique classes (`num_classes`) from the encoded labels.

* **One-Hot Encoding**:

  * Converts the integer-encoded labels into one-hot encoded format using `to_categorical()` for use in classification models.

* **Class and Data Size Validation**:

  * Ensures at least 2 samples and 2 unique classes exist before splitting the dataset.
    
  * If not, prompts the user to adjust the `window_size` or `overlap`, or provide a larger dataset.

* **Train-Test Split**:

  * Splits the dataset into training and testing sets using `train_test_split()`.
    
  * The split is stratified based on encoded class labels (`y_encoded`) to preserve class distribution.
    
  * A 70/30 train-test ratio is used with `random_state=42` for reproducibility.

* **Output Shapes Printed**:

  * Training and testing feature shapes (`X_train`, `X_test`).
    
  * One-hot encoded label shapes (`y_train`, `y_test`).
    
  * Testing set’s `window_ids` shape for traceability.
 
  2.**Feature Scaling** :Feature scaling is a crucial preprocessing step, especially when working with models like LSTMs (Long Short-Term Memory networks), which are sensitive to the magnitude and range of input values. In the given code, StandardScaler() is used to normalize the feature sets by removing the mean and scaling to unit variance. This is important because LSTMs rely on gradient-based optimization and internal gating mechanisms (like sigmoid and tanh activations) that can become unstable or ineffective when input values vary widely in scale. If one feature has values in the range of thousands while another varies only between 0 and 1, the larger feature can disproportionately influence weight updates, making training inefficient or unstable. Scaling all features to a similar range ensures that each contributes equally and the learning process converges more reliably.

  3.**Reshape Data for LSTM input**:Reshaping the feature data is necessary to prepare it for input into an LSTM model, which requires data in a 3D format: (samples, timesteps, features). The data is reshaped so that each sample contains 1 timestep and a fixed number of features (180), resulting in shapes like (num_samples, 1, num_features). This format tells the LSTM to treat each sample as a single moment in time with multiple sensor-derived statistics. Although timesteps = 1 doesn't allow the model to learn temporal patterns across multiple time steps, it ensures compatibility with the LSTM layer. To fully leverage LSTM’s ability to capture time-based dependencies, one would use timesteps > 1, where each sample represents a sequence of consecutive readings over time, enabling the model to learn how features evolve across multiple points in time.

# Model

| **Component**          | **Parameters / Settings**                                                       | **Purpose / Description**                                                                                                         |
|------------------------|----------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| **Model Type**         | `Sequential()`                                                                  | A linear stack of layers, ideal for feedforward architectures like LSTM.                                                          |
| **LSTM Layer**         | `units=100`<br>`input_shape=(timesteps, input_features)`<br>`activation='relu'` | Adds a Long Short-Term Memory layer to capture temporal dependencies in sensor data. ReLU activation aids in non-linear modeling. |
| **Dropout Layer**      | `rate=0.2`                                                                       | Prevents overfitting by randomly setting 20% of the LSTM outputs to zero during training.                                         |
| **Dense Output Layer** | `units=num_classes`<br>`activation='softmax'`                                   | Final classification layer. Outputs class probabilities using softmax activation.                                                 |
| **Loss Function**      | `'categorical_crossentropy'`                                                    | Suitable for multi-class classification using one-hot encoded targets.                                                            |
| **Optimizer**          | `'adam'`                                                                         | Adaptive Moment Estimation optimizer — effective and commonly used for deep learning.                                             |
| **Evaluation Metric**  | `'accuracy'`                                                                     | Measures the percentage of correctly predicted labels.                                                                            |
| **Epochs**             | `50`                                                                             | Number of passes over the training dataset.                                                                                       |
| **Batch Size**         | `32`                                                                             | Number of samples per gradient update. Smaller batches lead to more frequent updates.                                             |
| **Validation Split**   | `0.1`                                                                            | Uses 10% of training data for validation to monitor generalization performance.                                                   |
| **Verbose**            | `1`                                                                              | Enables real-time progress logging during training.                                                                               |

# Evaluation

**Classification Report**:The classification_report from sklearn.metrics provides a detailed breakdown of a classification model's performance across each class. It includes key evaluation metrics that help you understand how well the model is predicting each class — not just overall accuracy.

| Metric     | What it Measures                                                                 | Why It’s Useful                                                                                       |
|------------|-----------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| Precision  | Out of all predicted instances of a class, how many were actually correct?       | Helps understand how many false positives the model is making. High precision = fewer false alarms.   |
| Recall     | Out of all actual instances of a class, how many were correctly predicted?       | Indicates how well the model captures all relevant cases (true positives). High recall = few misses.  |
| F1-score   | Harmonic mean of precision and recall. Balances the two in a single metric.      | Useful when you want a balance between precision and recall, especially in imbalanced datasets.       |
| Support    | Number of actual occurrences of each class in the dataset.                       | Shows how many samples of each class were present in the test set — helps interpret other metrics.    |


* **Accuracy** measures the overall proportion of correct predictions across all classes.
  
* **Macro average** computes the unweighted mean of precision, recall, or F1-score across all classes, treating each class equally regardless of its size.

* **Weighted average** calculates the mean of precision, recall, or F1-score weighted by the number of true instances for each class, giving more influence to frequent classes.

# Possible resons behind failure

The extremely low accuracy (0.02 or 2%) in the LSTM model suggests that it struggled significantly to generalize meaningful patterns from the training data to
the test set. One of the most likely reasons is severe class imbalance, where one class (e.g., "Moves hand to target location") dominates the dataset with 
hundreds of examples, while other classes like "Hand at target location" or "Performs gesture" have very few. In such cases, models often become biased toward 
the majority class, failing to correctly identify rare behaviors, which leads to poor performance across minority classes.

Another possible reason is insufficient or noisy feature representation. If the statistical features extracted from sensor data windows (e.g., mean, std,skewness, etc.) do not adequately capture the behavioral patterns distinguishing BFRB from non-BFRB gestures, the LSTM model may struggle to learn meaningful temporal dependencies. Moreover, if the input data (windowed time series) is not standardized or has missing/inconsistent signals, this can negatively impact the 
model's ability to learn.


Lastly, the LSTM model itself may not be optimally configured or trained. A single LSTM layer with fixed units and default hyperparameters (e.g., dropout,
learning rate, number of epochs) might be insufficient for a complex, multi-class time series classification task. In particular, if the model was underfitting(not learning enough from data) or overfitting (memorizing noise), it would lead to poor generalization on the test set. Improvements might require hyperparameter tuning, feature selection, more balanced training data, or the addition of model complexity such as deeper layers or attention mechanisms.

   
# Possible alternate solution

* If you have **rich engineered features**, start with **XGBoost** or **LightGBM**.
  
* If you want to stick with deep learning but avoid LSTM's issues, try **1D CNNs** or **CNN-LSTM hybrids**.
  
* If you have enough data and resources, explore **Transformers or TCNs**.


 
    



