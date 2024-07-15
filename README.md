# Flight Anomaly Detection

This project uses machine learning techniques to detect anomalies in flight data. Specifically, it leverages an autoencoder neural network to identify unusual patterns and outliers in various flight parameters such as altitude, speed, and engine RPM.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)
- [Visualization](#visualization)

## Introduction

Anomaly detection in flight data is crucial for ensuring safety and maintaining the performance of aircraft. This project demonstrates how to use an autoencoder, a type of neural network, to identify anomalies in flight data. The approach involves training the autoencoder on normal flight data to learn an efficient representation and then using reconstruction errors to detect anomalies.

## Dataset

The dataset used in this project includes simulated flight data with various parameters:
- Timestamp
- Altitude (in feet)
- Speed (in knots)
- Heading (in degrees)
- Temperature (in °C)
- Pressure (in hPa)
- Vibration (in g)
- Latitude
- Longitude
- Roll (in degrees)
- Pitch (in degrees)
- Yaw (in degrees)
- Engine RPM
- Fuel Level (in percentage)
- Weather Conditions (categorical)

Anomalies are artificially introduced to simulate abnormal flight conditions.

## Requirements

- Python 3.8 or later
- Pandas
- NumPy
- TensorFlow
- scikit-learn
- Matplotlib
- Seaborn

## Usage

1. **Generate the dataset:**

   Create a Python script to generate the mock flight data (this script should be in a separate file). The dataset includes parameters such as timestamp, altitude, speed, heading, temperature, pressure, vibration, latitude, longitude, roll, pitch, yaw, engine RPM, fuel level, and weather conditions. Anomalies are introduced randomly in the dataset.

2. **Run the anomaly detection model:**

   Execute the provided script to preprocess the data, train the autoencoder model, and detect anomalies. The script will output a CSV file containing the flight data along with anomaly labels.

## Results

The output of the model is a CSV file (`enhanced_mock_flight_data_with_anomalies_autoencoder.csv`) that includes the original flight data and an additional column indicating anomalies (1 for anomalies, 0 for normal data).

## Visualization

The script generates visualizations to help understand the detected anomalies:
- **Altitude Anomalies:** A line plot of altitude over time with anomalies highlighted.
- **Speed Anomalies:** A line plot of speed over time with anomalies highlighted.
- **Engine RPM Anomalies:** A line plot of engine RPM over time with anomalies highlighted.

These plots use the seaborn library for better aesthetics and readability.
