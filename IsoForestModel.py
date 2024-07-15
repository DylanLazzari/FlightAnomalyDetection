import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
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

# Apply preprocessing
data_preprocessed = preprocessor.fit_transform(data)

# Generate column names for the resulting DataFrame
onehot_encoded_columns = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
processed_columns = numerical_features.tolist() + onehot_encoded_columns.tolist()
data_preprocessed = pd.DataFrame(data_preprocessed, columns=processed_columns)

# Split data into training and testing sets
X_train, X_test = train_test_split(data_preprocessed, test_size=0.2, random_state=42)

# Train Isolation Forest model
model = IsolationForest(contamination=0.02, random_state=42)
model.fit(X_train)

# Predict anomalies
data['Anomaly'] = model.predict(data_preprocessed)

# Convert -1 (anomalies) and 1 (normal) to binary format (1 for anomalies, 0 for normal)
data['Anomaly'] = data['Anomaly'].apply(lambda x: 1 if x == -1 else 0)

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
data.to_csv('enhanced_mock_flight_data_with_anomalies.csv', index=False)
