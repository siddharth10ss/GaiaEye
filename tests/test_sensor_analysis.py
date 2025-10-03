import pytest
import pandas as pd
from unittest.mock import patch

from src.sensor_analysis.analysis import analyze_sensor_data

@pytest.fixture
def mock_config():
    return {
        'thresholds': {
            'contamination_rate': 0.1
        }
    }

@pytest.fixture
def sample_sensor_data():
    data = {
        'timestamp': pd.to_datetime(['2023-01-01 12:00:00', '2023-01-01 12:01:00', '2023-01-01 12:02:00']),
        'temperature': [25.0, 26.0, 50.0],  # Anomaly
        'humidity': [50.0, 52.0, 55.0],
        'air_quality': [30.0, 32.0, 80.0]  # Anomaly
    }
    return pd.DataFrame(data)

def test_analyze_sensor_data_with_valid_input(mock_config, sample_sensor_data):
    """Test sensor data analysis with valid input."""
    with patch('src.sensor_analysis.analysis.pd.read_csv') as mock_read_csv, \
         patch('src.sensor_analysis.analysis.os.path.exists') as mock_exists:
        mock_read_csv.return_value = sample_sensor_data
        mock_exists.return_value = True

        anomalies = analyze_sensor_data('test.csv', mock_config)

        assert isinstance(anomalies, pd.DataFrame)
        assert not anomalies.empty
        assert 'anomaly' in anomalies.columns

def test_analyze_sensor_data_with_invalid_file(mock_config):
    """Test sensor data analysis with invalid file path."""
    anomalies = analyze_sensor_data('nonexistent.csv', mock_config)
    assert isinstance(anomalies, pd.DataFrame)
    assert anomalies.empty

def test_analyze_sensor_data_with_missing_columns(mock_config):
    """Test sensor data analysis with missing columns."""
    data = {
        'timestamp': pd.to_datetime(['2023-01-01 12:00:00']),
        'temperature': [25.0]
    }
    df = pd.DataFrame(data)

    with patch('src.sensor_analysis.analysis.pd.read_csv') as mock_read_csv, \
         patch('src.sensor_analysis.analysis.os.path.exists') as mock_exists:
        mock_read_csv.return_value = df
        mock_exists.return_value = True

        anomalies = analyze_sensor_data('test.csv', mock_config)

        assert isinstance(anomalies, pd.DataFrame)
        assert anomalies.empty
