# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

df=pd.read_csv("/kaggle/input/eeg-motor-imagery-bciciv-2a/BCICIV_2a_all_patients.csv")

eeg_data = df['EEG-Fz']
import matplotlib.pyplot as plt
plt.plot(eeg_data[400:600])
plt.show()

channels = df.drop(['patient' , 'time' , 'label' , 'epoch'], axis =1)
channels_train = channels[0:442200]
channels_test = channels[442200:]
print(channels_train.size/22)

split_data_train = []
num_splits = channels_train.size/(201*22)
rows = 201

for x in range(int(num_splits)):
    if x == 0:
        split = channels_train[(x)*rows  : (x+1)*rows ]
    else:
        split = channels_train[x*rows  : (x+1)*rows ]
    split_data_train.append(split)

import numpy as np 
from scipy.stats import kurtosis
from scipy.stats import skew

features = []
for approx_co in all_approx_co:
    feature_vector = []
    for approx in approx_co:
        feature_channel = []
        
        feature_channel.append(np.mean(approx))
        feature_channel.append(np.std(approx))
        feature_channel.append(kurtosis(approx))
        feature_channel.append(skew(approx))
        feature_channel.append(np.sum(approx**2))  # Energy
        feature_channel.append(-np.sum(approx * np.log(np.abs(approx) + 1e-10))) 
        
        feature_vector.append(np.array(feature_channel))
    features.append(feature_vector)

features_d = []
for detail_co in all_detail_co:
    feature_vector_d = []
    for detail in detail_co:
        feature_channel_d = []
        for d in detail:


            feature_channel_d.append(np.mean(approx))
            feature_channel_d.append(np.std(approx))
            feature_channel_d.append(kurtosis(approx))
            feature_channel_d.append(skew(approx))
            feature_channel_d.append(np.sum(approx**2))  # Energy
            feature_channel_d.append(-np.sum(approx * np.log(np.abs(approx) + 1e-10))) 

        feature_vector_d.append(np.array(feature_channel_d))
    features_d.append(feature_vector_d)

all_channel_features = []
for x in range(2200):
    channels_features = []
    for channel in range (22):
        approx_feat = features[x][channel]
        detail_feat = features_d[x][channel]
        channel_features = np.concatenate([approx_feat , detail_feat])
        
        channels_features.append(channel_features)
    
    combined_features = np.hstack(channels_features)
    
    all_channel_features.append(combined_features)
    

Y_train = []
for x in range(2200):
    Y_train.append(y[x*201])
    
Y_test = []
for x in range(248):
    Y_test.append(y[(x+2200)*201])

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
Y_train_encoded = label_encoder.fit_transform(Y_train)  # Convert to integers
print("Encoded Y_train shape:", Y_train_encoded.shape)

# One-hot encode the integer labels
Y_train_one_hot = tf.keras.utils.to_categorical(Y_train_encoded, num_classes=len(label_encoder.classes_))
print("One-hot encoded Y_train shape:", Y_train_one_hot.shape)


y = np.argmax(Y_train_one_hot, axis=1)


from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Assume all_channel_features and y are defined
scaler = StandardScaler()
X_scaled = scaler.fit_transform(all_channel_features)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

# Define the SVM model
svm = SVC(kernel='rbf', C=1.0, gamma='scale')

# Create the AdaBoost classifier
ada_boost = AdaBoostClassifier(base_estimator=svm, n_estimators=50, learning_rate=1.0)

# Fit the model
ada_boost.fit(X_train, y_train)

# Predict and evaluate
y_pred = ada_boost.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy of SVM with AdaBoost:", accuracy)

    