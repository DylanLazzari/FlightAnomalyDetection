
# Flight Data Anomaly Detection

This project aims to detect anomalies in flight data using machine learning models, specifically the Isolation Forest and One-Class SVM. The dataset includes features like altitude, speed, engine RPM, weather conditions, and geographic coordinates. The data is preprocessed, engineered, and visualized to identify patterns and outliers.

## Table of Contents
- [Installation](#installation)
- [Data Description](#data-description)
- [Preprocessing and Feature Engineering](#preprocessing-and-feature-engineering)
- [Anomaly Detection](#anomaly-detection)
- [Visualization](#visualization)
- [Results](#results)
- [Usage](#usage)
- [License](#license)

## Installation

Ensure you have Python 3.7+ installed. Install the required packages using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn geopandas shap plotly
```

## Data Description

The dataset, `enhanced_mock_flight_data.csv`, contains the following columns:
- `Timestamp`: Date and time of the record
- `Altitude`: Altitude of the flight
- `Speed`: Speed of the flight
- `Engine RPM`: Engine revolutions per minute
- `Latitude`: Geographic latitude
- `Longitude`: Geographic longitude
- `Weather Conditions`: Categorical weather conditions

## Preprocessing and Feature Engineering

1. **Timestamp Conversion**: Converts the `Timestamp` column to a datetime format.
2. **Lag Features**: Creates lag features for `Altitude`, `Speed`, and `Engine RPM` to capture previous time step values.
3. **Rolling Standard Deviation**: Calculates the rolling standard deviation over a window of 3 time steps for `Altitude`, `Speed`, and `Engine RPM`.
4. **Missing Values Imputation**: Fills missing values using K-Nearest Neighbors Imputer (KNN Imputer).
5. **Data Transformation**: Applies StandardScaler to numerical features and OneHotEncoder to categorical features.

## Anomaly Detection

Two models are used to detect anomalies:
1. **Isolation Forest**:
    - `contamination=0.02`: Expects about 2% of the data to be anomalies.
    - Outputs `-1` for anomalies and `1` for normal points.
2. **One-Class SVM**:
    - `gamma='auto'`, `nu=0.02`: Parameters for the SVM model.
    - Outputs `-1` for anomalies and `1` for normal points.

The models are trained on 80% of the preprocessed data and tested on 20%.

## Visualization

Interactive visualizations are created using Plotly:
- **Flight Paths**: Normal points and anomalies are plotted on a world map.
- **Time Series**: Anomalies in `Altitude`, `Speed`, and `Engine RPM` over time are visualized.

## Results

The results are saved to `enhanced_mock_flight_data_with_anomalies_advanced.csv`, including the anomaly labels from both models.

## Usage

Run the provided script to preprocess the data, detect anomalies, and visualize the results. Ensure your dataset is named `enhanced_mock_flight_data.csv` and placed in the same directory as the script.

```bash
python flight_data_anomaly_detection.py
```

## License

This project is licensed under the MIT License.
