import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import shap

# Load the data from a CSV file into a pandas DataFrame
data = pd.read_csv('enhanced_mock_flight_data.csv')

# Convert 'Timestamp' column to datetime format for better handling of time-related data
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Feature Engineering: Add rolling averages and differences to capture trends and sudden changes
# This step calculates how much a value has changed compared to the previous value (difference)
data['Altitude_diff'] = data['Altitude'].diff()  # Difference in altitude
data['Speed_diff'] = data['Speed'].diff()  # Difference in speed
data['Engine_RPM_diff'] = data['Engine RPM'].diff()  # Difference in engine RPM

# Calculate rolling averages for smoothing the data
# Rolling averages help to smooth out fluctuations and highlight trends in the data
data['Altitude_roll_mean'] = data['Altitude'].rolling(window=3).mean()
data['Speed_roll_mean'] = data['Speed'].rolling(window=3).mean()
data['Engine_RPM_roll_mean'] = data['Engine RPM'].rolling(window=3).mean()

# Handle missing values created by differencing and rolling operations using backward fill
# Fill in the gaps created by the new features so the data is complete
data.bfill(inplace=True)

# Identify the categorical (non-numeric) and numerical (numeric) features to be processed
categorical_features = ['Weather Conditions']
numerical_features = data.columns.difference(categorical_features + ['Timestamp', 'Latitude', 'Longitude'])

# Define the preprocessing step for numerical features: StandardScaler
# StandardScaler standardizes features by removing the mean and scaling to unit variance (making them comparable)
numerical_transformer = StandardScaler()

# Define the preprocessing step for categorical features: OneHotEncoder
# OneHotEncoder converts categorical data (like weather conditions) into a numerical format
# drop='first' is used to avoid redundant information
categorical_transformer = OneHotEncoder(drop='first')

# Combine preprocessing steps for numerical and categorical features
# This step ensures that the right transformation is applied to each type of feature
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),  # Apply StandardScaler to numerical features
        ('cat', categorical_transformer, categorical_features)  # Apply OneHotEncoder to categorical features
    ])

# Fit the preprocessor to the data and transform the data accordingly
data_preprocessed = preprocessor.fit_transform(data)

# Generate column names for the resulting DataFrame after preprocessing
# Get the new column names after one-hot encoding the categorical features
onehot_encoded_columns = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)

# Combine numerical feature names and one-hot encoded categorical feature names
processed_columns = numerical_features.tolist() + onehot_encoded_columns.tolist()

# Create a new DataFrame with the preprocessed data and the generated column names
data_preprocessed = pd.DataFrame(data_preprocessed, columns=processed_columns)

# Split the preprocessed data into training (80%) and testing (20%) sets
X_train, X_test = train_test_split(data_preprocessed, test_size=0.2, random_state=42)

# Initialize the Isolation Forest model for anomaly detection
# Isolation Forest helps to identify outliers or anomalies in the data
# contamination=0.02 means we expect about 2% of the data to be anomalies
model = IsolationForest(contamination=0.02, random_state=42)

# Train the Isolation Forest model on the training data
model.fit(X_train)

# Predict anomalies on the entire preprocessed dataset
# The model outputs -1 for anomalies and 1 for normal points
data['Anomaly'] = model.predict(data_preprocessed)

# Convert the prediction results to binary format: 1 for anomalies, 0 for normal points
data['Anomaly'] = data['Anomaly'].apply(lambda x: 1 if x == -1 else 0)

# Geospatial Analysis: Visualize flight paths and anomalies on a map
world = gpd.read_file('')

# Create a GeoDataFrame with flight data, using latitude and longitude for geospatial analysis
gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.Longitude, data.Latitude))

# Plot world map and overlay flight paths
fig, ax = plt.subplots(figsize=(15, 15))
world.plot(ax=ax, color='lightgrey')  # Plot the world map
gdf.plot(ax=ax, marker='o', color='blue', markersize=5, label='Normal')  # Plot normal flight paths
gdf[gdf['Anomaly'] == 1].plot(ax=ax, marker='o', color='red', markersize=5, label='Anomaly')  # Plot anomalies
plt.legend()
plt.title('Flight Paths with Anomalies')
plt.show()

# Detailed Analysis: Count the total number of records and anomalies detected
anomaly_count = data['Anomaly'].sum()
total_count = data.shape[0]
print(f'Total records: {total_count}, Anomalies detected: {anomaly_count}')

# Visualization of anomalies in Altitude over time
plt.figure(figsize=(14, 7))  # Set the figure size
sns.lineplot(x='Timestamp', y='Altitude', data=data, label='Altitude')  # Plot Altitude over time
sns.scatterplot(x='Timestamp', y='Altitude', data=data[data['Anomaly'] == 1], color='red', label='Anomalies')  # Highlight anomalies
plt.xlabel('Timestamp')
plt.ylabel('Altitude')
plt.legend()
plt.title('Altitude Anomalies')
plt.show()

# Visualization of anomalies in Speed over time
plt.figure(figsize=(14, 7))  # Set the figure size
sns.lineplot(x='Timestamp', y='Speed', data=data, label='Speed')  # Plot Speed over time
sns.scatterplot(x='Timestamp', y='Speed', data=data[data['Anomaly'] == 1], color='red', label='Anomalies')  # Highlight anomalies
plt.xlabel('Timestamp')
plt.ylabel('Speed')
plt.legend()
plt.title('Speed Anomalies')
plt.show()

# Visualization of anomalies in Engine RPM over time
plt.figure(figsize=(14, 7))  # Set the figure size
sns.lineplot(x='Timestamp', y='Engine RPM', data=data, label='Engine RPM')  # Plot Engine RPM over time
sns.scatterplot(x='Timestamp', y='Engine RPM', data=data[data['Anomaly'] == 1], color='red', label='Anomalies')  # Highlight anomalies
plt.xlabel('Timestamp')
plt.ylabel('Engine RPM')
plt.legend()
plt.title('Engine RPM Anomalies')
plt.show()

# Use SHAP to explain anomalies
# SHAP (SHapley Additive exPlanations) helps explain the output of machine learning models
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

# Summary plot of SHAP values
# This plot shows the impact of each feature on the model's output
shap.summary_plot(shap_values, X_train)

# Save the results with anomalies to a new CSV file for further analysis or reporting
data.to_csv('enhanced_mock_flight_data_with_anomalies.csv', index=False)
