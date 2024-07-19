import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('enhanced_mock_flight_data.csv')

# Convert timestamp to datetime
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Handle categorical variables (weather conditions)
categorical_features = ['Weather Conditions']
numerical_features = data.columns.difference(categorical_features + ['Timestamp'])

# Define preprocessing for numerical features (scaling)
numerical_transformer = StandardScaler()

# Define preprocessing for categorical features (one-hot encoding)
categorical_transformer = OneHotEncoder(drop='first')

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Fit the preprocessor to the data and transform the data accordingly
# 'Fitting' means calculating the necessary parameters from the data
data_preprocessed = preprocessor.fit_transform(data)

# Generate column names for the resulting DataFrame
onehot_encoded_columns = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
processed_columns = numerical_features.tolist() + onehot_encoded_columns.tolist()
data_preprocessed = pd.DataFrame(data_preprocessed, columns=processed_columns) # This is creating your new table with the preprocessed data names.

# Split data into training and testing sets
X_train, X_test = train_test_split(data_preprocessed, test_size=0.2, random_state=42)

# Define the autoencoder model
input_dim = X_train.shape[1]
encoding_dim = 14  # Number of neurons in the hidden layer

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation="relu")(input_layer)
decoder = Dense(input_dim, activation="sigmoid")(encoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)

# Compile the model
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = autoencoder.fit(X_train, X_train,
                          epochs=50,
                          batch_size=32,
                          validation_split=0.2,
                          shuffle=True)

# Predict the reconstruction for the entire dataset
data_pred = autoencoder.predict(data_preprocessed)

# Calculate reconstruction error
mse = np.mean(np.power(data_preprocessed - data_pred, 2), axis=1)

# Define a threshold for anomaly detection
threshold = np.percentile(mse, 95)

# Label data points as anomalies if reconstruction error exceeds the threshold
data['Anomaly'] = np.where(mse > threshold, 1, 0)

# Visualization of anomalies in Altitude
plt.figure(figsize=(14, 7))
sns.lineplot(x='Timestamp', y='Altitude', data=data, label='Altitude')
sns.scatterplot(x='Timestamp', y='Altitude', data=data[data['Anomaly'] == 1], color='red', label='Anomalies')
plt.xlabel('Timestamp')
plt.ylabel('Altitude')
plt.legend()
plt.title('Altitude Anomalies')
plt.show()

# Visualization of anomalies in Speed
plt.figure(figsize=(14, 7))
sns.lineplot(x='Timestamp', y='Speed', data=data, label='Speed')
sns.scatterplot(x='Timestamp', y='Speed', data=data[data['Anomaly'] == 1], color='red', label='Anomalies')
plt.xlabel('Timestamp')
plt.ylabel('Speed')
plt.legend()
plt.title('Speed Anomalies')
plt.show()

# Visualization of anomalies in Engine RPM
plt.figure(figsize=(14, 7))
sns.lineplot(x='Timestamp', y='Engine RPM', data=data, label='Engine RPM')
sns.scatterplot(x='Timestamp', y='Engine RPM', data=data[data['Anomaly'] == 1], color='red', label='Anomalies')
plt.xlabel('Timestamp')
plt.ylabel('Engine RPM')
plt.legend()
plt.title('Engine RPM Anomalies')
plt.show()

# Save results with anomalies
data.to_csv('enhanced_mock_flight_data_with_anomalies_autoencoder.csv', index=False)
