Repository Description
This repository contains code for analyzing EEG data related to motor imagery tasks using machine learning techniques. The dataset consists of EEG recordings from multiple patients, with channels corresponding to various motor imagery tasks such as left hand, right hand, foot, and tongue movements.

Key Features:
Data Loading and Preprocessing:

Loads the EEG dataset and visualizes the data.
Segments the data into training and test sets.
Feature Extraction:

Implements signal processing techniques to extract statistical features from the EEG signals, including mean, standard deviation, kurtosis, skewness, energy, and entropy for both approximate and detail coefficients.
Label Encoding and One-Hot Encoding:

Encodes the labels of the motor imagery tasks for use in machine learning models.
Machine Learning Model:

Utilizes a Support Vector Machine (SVM) classifier combined with AdaBoost for improved prediction accuracy on the EEG feature data.
Model Evaluation:

Evaluates the model's performance by calculating accuracy on a test set.
Technologies Used:
Python 3
NumPy, Pandas for data manipulation
Matplotlib for data visualization
Scikit-learn for machine learning and preprocessing
TensorFlow for one-hot encoding
Dataset:
The dataset used in this analysis can be accessed at BCICIV 2a Motor Imagery Dataset.

How to Run:
To run the code, ensure you have the necessary packages installed in your Python environment, then execute the script to load the dataset, extract features, train the model, and evaluate its performance.

Feel free to explore and modify the code for further analysis or experimentation with EEG data!
