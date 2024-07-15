import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Parameters for the dataset
num_records = 1000
start_time = datetime.now()
time_interval = timedelta(seconds=1)

# Generating the mock data
timestamps = [start_time + i * time_interval for i in range(num_records)]
altitude = np.random.normal(loc=30000, scale=1000, size=num_records).tolist()  # Mean at 30,000 feet
speed = np.random.normal(loc=500, scale=50, size=num_records).tolist()  # Mean at 500 knots
heading = np.random.uniform(low=0, high=360, size=num_records).tolist()  # 0 to 360 degrees
temperature = np.random.normal(loc=-50, scale=5, size=num_records).tolist()  # Mean at -50°C
pressure = np.random.normal(loc=1013, scale=10, size=num_records).tolist()  # Mean at 1013 hPa
vibration = np.random.normal(loc=0.5, scale=0.1, size=num_records).tolist()  # Mean at 0.5g
latitude = np.random.uniform(low=-90, high=90, size=num_records).tolist()  # -90 to 90 degrees
longitude = np.random.uniform(low=-180, high=180, size=num_records).tolist()  # -180 to 180 degrees
roll = np.random.uniform(low=-180, high=180, size=num_records).tolist()  # -180 to 180 degrees
pitch = np.random.uniform(low=-90, high=90, size=num_records).tolist()  # -90 to 90 degrees
yaw = np.random.uniform(low=0, high=360, size=num_records).tolist()  # 0 to 360 degrees
engine_rpm = np.random.normal(loc=2000, scale=200, size=num_records).tolist()  # Mean at 2000 RPM
fuel_level = np.random.uniform(low=0, high=100, size=num_records).tolist()  # 0 to 100 percent
weather_conditions = np.random.choice(['Clear', 'Cloudy', 'Stormy'], size=num_records).tolist()

# Introducing anomalies
for i in range(20):  # Introduce 20 anomalies
    index = random.randint(0, num_records - 1)
    altitude[index] = np.random.uniform(low=10000, high=50000)  # Anomalous altitude
    speed[index] = np.random.uniform(low=100, high=700)  # Anomalous speed
    heading[index] = np.random.uniform(low=0, high=360)  # Random heading
    temperature[index] = np.random.uniform(low=-80, high=0)  # Anomalous temperature
    pressure[index] = np.random.uniform(low=900, high=1100)  # Anomalous pressure
    vibration[index] = np.random.uniform(low=0.1, high=1.0)  # Anomalous vibration
    roll[index] = np.random.uniform(low=-180, high=180)  # Anomalous roll
    pitch[index] = np.random.uniform(low=-90, high=90)  # Anomalous pitch
    yaw[index] = np.random.uniform(low=0, high=360)  # Anomalous yaw
    engine_rpm[index] = np.random.uniform(low=1000, high=4000)  # Anomalous engine RPM
    fuel_level[index] = np.random.uniform(low=0, high=100)  # Random fuel level
    weather_conditions[index] = 'Stormy'  # Anomalous weather condition

# Creating the DataFrame
data = pd.DataFrame({
    'Timestamp': timestamps,
    'Altitude': altitude,
    'Speed': speed,
    'Heading': heading,
    'Temperature': temperature,
    'Pressure': pressure,
    'Vibration': vibration,
    'Latitude': latitude,
    'Longitude': longitude,
    'Roll': roll,
    'Pitch': pitch,
    'Yaw': yaw,
    'Engine RPM': engine_rpm,
    'Fuel Level': fuel_level,
    'Weather Conditions': weather_conditions
})

# Save to CSV
data.to_csv('enhanced_mock_flight_data.csv', index=False)

print("Enhanced mock flight data generated and saved to 'enhanced_mock_flight_data.csv'")
