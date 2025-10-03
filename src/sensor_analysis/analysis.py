import pandas as pd
from sklearn.ensemble import IsolationForest
import logging
import os
from src.config import config

class SensorAnalysisPipeline:
    """
    A pipeline for analyzing sensor data to detect anomalies using an Isolation Forest model.
    """
    def __init__(self):
        """
        Initializes the sensor analysis pipeline with configuration settings.
        """
        sensor_config = config.get('sensor_analysis', {})
        self.contamination_rate = sensor_config.get('contamination_rate', 0.1)
        self.required_columns = ['temperature', 'humidity', 'air_quality']
    
    def analyze(self, csv_file_path: str) -> pd.DataFrame:
        """
        Analyzes sensor data from a CSV file to detect anomalies.

        Args:
            csv_file_path (str): The path to the CSV file containing sensor data.

        Returns:
            pd.DataFrame: A DataFrame containing the detected anomalies.
        """
        if not os.path.exists(csv_file_path):
            logging.error(f"Sensor data file not found: {csv_file_path}")
            return pd.DataFrame()

        try:
            data = pd.read_csv(csv_file_path)
        except Exception as e:
            logging.error(f"Error reading sensor data file {csv_file_path}: {e}")
            return pd.DataFrame()

        missing_columns = [col for col in self.required_columns if col not in data.columns]
        if missing_columns:
            logging.error(f"Missing required columns in sensor data: {missing_columns}")
            return pd.DataFrame()

        if data.empty:
            logging.warning("Sensor data is empty after loading.")
            return pd.DataFrame()

        X = data[self.required_columns]
        if X.isnull().values.any():
            logging.warning("Sensor data contains NaN values. Dropping rows with NaN values.")
            X = X.dropna()
            if X.empty:
                logging.error("No valid sensor data after dropping NaN values.")
                return pd.DataFrame()

        try:
            model = IsolationForest(contamination=self.contamination_rate, random_state=42)
            model.fit(X)
            data['anomaly'] = model.predict(X)
            logging.info(f"Anomaly detection complete. Found {sum(data['anomaly'] == -1)} anomalies.")
        except Exception as e:
            logging.error(f"Error during anomaly detection: {e}")
            return pd.DataFrame()

        return data[data['anomaly'] == -1]

if __name__ == '__main__':
    # This part is for direct testing of the script
    
    test_csv_path = 'data/sensor/sensor_data.csv'

    sensor_pipeline = SensorAnalysisPipeline()
    anomalies = sensor_pipeline.analyze(test_csv_path)

    print("--- Sensor Data Analysis Results ---")
    if not anomalies.empty:
        print("Detected anomalies:")
        for index, row in anomalies.iterrows():
            print(f"- Timestamp: {row['timestamp']}, Temp: {row['temperature']}, Humidity: {row['humidity']}, Air Quality: {row['air_quality']}")
    else:
        print("No anomalies detected.")