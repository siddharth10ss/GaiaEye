import pandas as pd
from sklearn.ensemble import IsolationForest
import logging
from typing import Dict, Any, Optional
import os

def analyze_sensor_data(csv_file_path: str, config: Optional[Dict[Any, Any]] = None) -> pd.DataFrame:
    """
    Analyzes sensor data from a CSV file to detect anomalies using an Isolation Forest model.

    Args:
        csv_file_path (str): The path to the CSV file containing sensor data.
        config (dict, optional): Configuration dictionary.

    Returns:
        pd.DataFrame: A DataFrame containing the detected anomalies.
    """
    if config is None:
        config = {}
    
    # Get configuration values with defaults
    contamination_rate = config.get('thresholds', {}).get('contamination_rate', 0.1)
    
    # Validate file exists
    if not os.path.exists(csv_file_path):
        logging.error(f"Sensor data file not found: {csv_file_path}")
        return pd.DataFrame()
    
    # Load the sensor data
    try:
        data = pd.read_csv(csv_file_path)
        logging.info(f"Loaded sensor data from {csv_file_path} with {len(data)} records")
    except FileNotFoundError:
        logging.error(f"Error: The file {csv_file_path} was not found.")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        logging.error(f"Error: The file {csv_file_path} is empty.")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error reading sensor data file {csv_file_path}: {e}")
        return pd.DataFrame()

    # Validate required columns exist
    required_columns = ['temperature', 'humidity', 'air_quality']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logging.error(f"Missing required columns in sensor data: {missing_columns}")
        return pd.DataFrame()
    
    # Check for empty data after column validation
    if data.empty:
        logging.warning("Sensor data is empty after loading")
        return pd.DataFrame()
    
    # Prepare the data for the model
    try:
        X = data[required_columns]
        
        # Check for NaN values
        if X.isnull().values.any():
            logging.warning("Sensor data contains NaN values. Dropping rows with NaN values.")
            X = X.dropna()
            if X.empty:
                logging.error("No valid sensor data after dropping NaN values")
                return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error preparing sensor data for analysis: {e}")
        return pd.DataFrame()

    try:
        # Initialize and train the Isolation Forest model
        model = IsolationForest(contamination=contamination_rate, random_state=42)
        model.fit(X)
        logging.info(f"Trained Isolation Forest model with contamination rate: {contamination_rate}")

        # Predict the anomalies (-1 for anomalies, 1 for inliers)
        data['anomaly'] = model.predict(X)
        logging.info(f"Anomaly detection complete. Found {sum(data['anomaly'] == -1)} anomalies.")
    except Exception as e:
        logging.error(f"Error during anomaly detection: {e}")
        return pd.DataFrame()

    # Return the anomalous data
    anomalies = data[data['anomaly'] == -1]
    return anomalies


if __name__ == '__main__':
    # This part is for direct testing of the script
    import yaml
    
    # Load config for testing
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except:
        config = {}
    
    test_csv_path = 'data/sensor/sensor_data.csv'
    anomalies = analyze_sensor_data(test_csv_path, config)

    print("--- Sensor Data Analysis Results ---")
    if not anomalies.empty:
        print("Detected anomalies:")
        for index, row in anomalies.iterrows():
            print(f"- Timestamp: {row['timestamp']}, Temp: {row['temperature']}, Humidity: {row['humidity']}, Air Quality: {row['air_quality']}")
    else:
        print("No anomalies detected.")
