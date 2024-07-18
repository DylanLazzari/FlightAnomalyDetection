import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import shap
import plotly.express as px
import plotly.graph_objects as go

# Load the flight data from a CSV file into a pandas DataFrame (table)
data = pd.read_csv('enhanced_mock_flight_data.csv')

# Convert the 'Timestamp' column to a date and time format for better handling of time-related data
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Create additional features (columns) to help capture patterns over time
# Lag features represent values from the previous time step (e.g., previous altitude, speed, and engine RPM)
data['Altitude_lag1'] = data['Altitude'].shift(1)
data['Speed_lag1'] = data['Speed'].shift(1)
data['Engine_RPM_lag1'] = data['Engine RPM'].shift(1)

# Calculate the standard deviation over a window of 3 time steps
# This helps to understand how much variation or 'noise' there is in the data over short periods
data['Altitude_roll_std'] = data['Altitude'].rolling(window=3).std()
data['Speed_roll_std'] = data['Speed'].rolling(window=3).std()
data['Engine_RPM_roll_std'] = data['Engine RPM'].rolling(window=3).std()

# Fill in any missing values created by the new features
# We use a method called K-Nearest Neighbors Imputer (KNN Imputer) which fills in missing values based on nearby data points
# First, we exclude non-numeric columns because they cannot be processed by the imputer
numeric_data = data.drop(columns=['Timestamp', 'Latitude', 'Longitude', 'Weather Conditions'])

# Apply the KNN Imputer to fill in missing values
imputer = KNNImputer(n_neighbors=3)
numeric_data.iloc[:, :] = imputer.fit_transform(numeric_data)

# After filling in missing values, restore the non-numeric columns back to the original table
data.update(numeric_data)

# Identify which columns are categorical (describing categories or groups) and which are numerical (describing numbers)
categorical_features = ['Weather Conditions']
numerical_features = numeric_data.columns.difference(categorical_features + ['Latitude', 'Longitude'])

# Define the preprocessing steps for numerical features: StandardScaler
# StandardScaler standardizes features by removing the mean and scaling to unit variance (making them comparable)
numerical_transformer = StandardScaler()

# Define the preprocessing steps for categorical features: OneHotEncoder
# OneHotEncoder converts categorical data (like weather conditions) into a numerical format
# drop='first' is used to avoid redundant information (only one column needed per category)
categorical_transformer = OneHotEncoder(drop='first')

# Combine preprocessing steps for numerical and categorical features
# This step ensures that the right transformation is applied to each type of feature
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Fit the preprocessor to the data and transform the data accordingly
data_preprocessed = preprocessor.fit_transform(data)

# Generate column names for the resulting table after preprocessing
# Get the new column names after one-hot encoding the categorical features
onehot_encoded_columns = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
processed_columns = numerical_features.tolist() + onehot_encoded_columns.tolist()

# Create a new table with the preprocessed data and the generated column names
data_preprocessed = pd.DataFrame(data_preprocessed, columns=processed_columns)

# Split the preprocessed data into training (80%) and testing (20%) sets
# This step helps to train the model on one part of the data and test its performance on another part
X_train, X_test = train_test_split(data_preprocessed, test_size=0.2, random_state=42)

# Initialize the Isolation Forest model for anomaly detection
# Isolation Forest helps to identify outliers or anomalies in the data
# contamination=0.02 means we expect about 2% of the data to be anomalies
model_if = IsolationForest(contamination=0.02, random_state=42)
model_if.fit(X_train)

# Initialize One-Class SVM for comparison
# One-Class SVM is another method for anomaly detection
model_ocsvm = OneClassSVM(gamma='auto', nu=0.02)
model_ocsvm.fit(X_train)

# Predict anomalies on the entire preprocessed dataset
# The model outputs -1 for anomalies and 1 for normal points
data['Anomaly_IF'] = model_if.predict(data_preprocessed)
data['Anomaly_OCSVM'] = model_ocsvm.predict(data_preprocessed)

# Convert the prediction results to binary format: 1 for anomalies, 0 for normal points
data['Anomaly_IF'] = data['Anomaly_IF'].apply(lambda x: 1 if x == -1 else 0)
data['Anomaly_OCSVM'] = data['Anomaly_OCSVM'].apply(lambda x: 1 if x == -1 else 0)

# Visualize flight paths and anomalies on an interactive map
# We use Plotly for interactive visualizations
fig = go.Figure()

# Plot normal flight paths
fig.add_trace(go.Scattergeo(
    lon=data[data['Anomaly_IF'] == 0]['Longitude'],
    lat=data[data['Anomaly_IF'] == 0]['Latitude'],
    mode='markers',
    marker=dict(color='blue', size=5),
    name='Normal'
))

# Plot anomalies detected by Isolation Forest
fig.add_trace(go.Scattergeo(
    lon=data[data['Anomaly_IF'] == 1]['Longitude'],
    lat=data[data['Anomaly_IF'] == 1]['Latitude'],
    mode='markers',
    marker=dict(color='red', size=5),
    name='Anomaly IF'
))

# Plot anomalies detected by One-Class SVM
fig.add_trace(go.Scattergeo(
    lon=data[data['Anomaly_OCSVM'] == 1]['Longitude'],
    lat=data[data['Anomaly_OCSVM'] == 1]['Latitude'],
    mode='markers',
    marker=dict(color='orange', size=5),
    name='Anomaly OCSVM'
))

fig.update_layout(
    title='Flight Paths with Anomalies',
    geo_scope='world'  # Limit the map to the world view
)
fig.show()

# Count the total number of records and anomalies detected by each model
anomaly_count_if = data['Anomaly_IF'].sum()
anomaly_count_ocsvm = data['Anomaly_OCSVM'].sum()
total_count = data.shape[0]
print(f'Total records: {total_count}, Anomalies detected by IF: {anomaly_count_if}, Anomalies detected by OCSVM: {anomaly_count_ocsvm}')

# Visualization of anomalies in Altitude over time using Plotly for interactivity
fig = px.line(data, x='Timestamp', y='Altitude', title='Altitude Anomalies')
fig.add_scatter(x=data[data['Anomaly_IF'] == 1]['Timestamp'], y=data[data['Anomaly_IF'] == 1]['Altitude'], mode='markers', name='Anomaly IF', marker=dict(color='red'))
fig.add_scatter(x=data[data['Anomaly_OCSVM'] == 1]['Timestamp'], y=data[data['Anomaly_OCSVM'] == 1]['Altitude'], mode='markers', name='Anomaly OCSVM', marker=dict(color='orange'))
fig.show()

# Visualization of anomalies in Speed over time using Plotly for interactivity
fig = px.line(data, x='Timestamp', y='Speed', title='Speed Anomalies')
fig.add_scatter(x=data[data['Anomaly_IF'] == 1]['Timestamp'], y=data[data['Anomaly_IF'] == 1]['Speed'], mode='markers', name='Anomaly IF', marker=dict(color='red'))
fig.add_scatter(x=data[data['Anomaly_OCSVM'] == 1]['Timestamp'], y=data[data['Anomaly_OCSVM'] == 1]['Speed'], mode='markers', name='Anomaly OCSVM', marker=dict(color='orange'))
fig.show()

# Visualization of anomalies in Engine RPM over time using Plotly for interactivity
fig = px.line(data, x='Timestamp', y='Engine RPM', title='Engine RPM Anomalies')
fig.add_scatter(x=data[data['Anomaly_IF'] == 1]['Timestamp'], y=data[data['Anomaly_IF'] == 1]['Engine RPM'], mode='markers', name='Anomaly IF', marker=dict(color='red'))
fig.add_scatter(x=data[data['Anomaly_OCSVM'] == 1]['Timestamp'], y=data[data['Anomaly_OCSVM'] == 1]['Engine RPM'], mode='markers', name='Anomaly OCSVM', marker=dict(color='orange'))
fig.show()

# Use SHAP to explain anomalies
# SHAP (SHapley Additive exPlanations) helps explain the output of machine learning models
# We use SHAP to understand which features are most important in detecting anomalies
explainer_if = shap.TreeExplainer(model_if)
shap_values_if = explainer_if.shap_values(X_train)

# Summary plot of SHAP values for Isolation Forest
shap.summary_plot(shap_values_if, X_train, plot_type="bar", title="SHAP Summary Plot for Isolation Forest")

# Save the results with anomalies to a new CSV file for further analysis or reporting
data.to_csv('enhanced_mock_flight_data_with_anomalies_advanced.csv', index=False)
