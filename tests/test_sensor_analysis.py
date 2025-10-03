import pytest
import pandas as pd
from unittest.mock import patch

from src.sensor_analysis.analysis import SensorAnalysisPipeline

@pytest.fixture
def mock_config():
    """Fixture to mock the centralized config for sensor analysis."""
    with patch('src.sensor_analysis.analysis.config', {
        'sensor_analysis': {
            'contamination_rate': 0.1
        }
    }) as mock_cfg:
        yield mock_cfg

@pytest.fixture
def sample_sensor_data():
    """Fixture to provide sample sensor data with clear anomalies."""
    data = {
        'timestamp': pd.to_datetime(['2023-01-01 12:00:00', '2023-01-01 12:01:00', '2023-01-01 12:02:00']),
        'temperature': [25.0, 26.0, 90.0],  # Clear anomaly
        'humidity': [50.0, 52.0, 55.0],
        'air_quality': [30.0, 32.0, 150.0] # Clear anomaly
    }
    return pd.DataFrame(data)

def test_analyze_sensor_data_with_valid_input(mock_config, sample_sensor_data):
    """Test sensor data analysis with valid input using the pipeline."""
    with patch('pandas.read_csv', return_value=sample_sensor_data), \
         patch('os.path.exists', return_value=True):

        sensor_pipeline = SensorAnalysisPipeline()
        anomalies = sensor_pipeline.analyze('dummy_sensor_data.csv')

        assert isinstance(anomalies, pd.DataFrame)
        assert not anomalies.empty
        assert 'anomaly' in anomalies.columns
        # With a contamination of 0.1, the model should flag the two outlier points.
        # Depending on the model's behavior, it might find one or both. We check for > 0.
        assert len(anomalies) > 0

def test_analyze_sensor_data_with_invalid_file(mock_config):
    """Test the pipeline with an invalid file path."""
    with patch('os.path.exists', return_value=False):
        sensor_pipeline = SensorAnalysisPipeline()
        anomalies = sensor_pipeline.analyze('nonexistent.csv')

        assert isinstance(anomalies, pd.DataFrame)
        assert anomalies.empty

def test_analyze_sensor_data_with_missing_columns(mock_config):
    """Test the pipeline with data missing required columns."""
    data = {
        'timestamp': pd.to_datetime(['2023-01-01 12:00:00']),
        'temperature': [25.0]
    }
    df = pd.DataFrame(data)

    with patch('pandas.read_csv', return_value=df), \
         patch('os.path.exists', return_value=True):

        sensor_pipeline = SensorAnalysisPipeline()
        anomalies = sensor_pipeline.analyze('dummy_missing_cols.csv')

        assert isinstance(anomalies, pd.DataFrame)
        assert anomalies.empty