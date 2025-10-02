import pandas as pd
from sklearn.ensemble import IsolationForest

def analyze_sensor_data(csv_file_path):
    """
    Analyzes sensor data from a CSV file to detect anomalies using an Isolation Forest model.

    Args:
        csv_file_path (str): The path to the CSV file containing sensor data.

    Returns:
        pd.DataFrame: A DataFrame containing the detected anomalies.
    """
    # Load the sensor data
    try:
        data = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: The file {csv_file_path} was not found.")
        return pd.DataFrame()

    # Prepare the data for the model
    X = data[['temperature', 'humidity', 'air_quality']]

    # Initialize and train the Isolation Forest model
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X)

    # Predict the anomalies (-1 for anomalies, 1 for inliers)
    data['anomaly'] = model.predict(X)

    # Return the anomalous data
    return data[data['anomaly'] == -1]


if __name__ == '__main__':
    # This part is for direct testing of the script
    test_csv_path = 'data/sensor/sensor_data.csv'
    anomalies = analyze_sensor_data(test_csv_path)

    print("--- Sensor Data Analysis Results ---")
    if not anomalies.empty:
        print("Detected anomalies:")
        for index, row in anomalies.iterrows():
            print(f"- Timestamp: {row['timestamp']}, Temp: {row['temperature']}, Humidity: {row['humidity']}, Air Quality: {row['air_quality']}")
    else:
        print("No anomalies detected.")