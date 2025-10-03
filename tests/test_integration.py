import pytest
import pandas as pd
from unittest.mock import patch

from src.integration.reporting import generate_consolidated_report, correlate_multimodal_data

@pytest.fixture
def sample_video_results():
    """Fixture for sample video analysis results."""
    return {'fire': 0.9, 'smoke': 0.8}

@pytest.fixture
def sample_audio_results():
    """Fixture for sample audio analysis results."""
    return {'Fire alarm': 0.85, 'Fire': 0.7}

@pytest.fixture
def sample_sensor_anomalies():
    """Fixture for sample sensor anomaly data."""
    data = {
        'timestamp': pd.to_datetime(['2023-01-01 12:02:00']),
        'temperature': [50.0],
        'humidity': [55.0],
        'air_quality': [80.0]
    }
    return pd.DataFrame(data)

@pytest.fixture
def no_sensor_anomalies():
    """Fixture for an empty sensor anomaly DataFrame."""
    return pd.DataFrame()

@pytest.fixture
def mock_config():
    """Fixture to mock the centralized config for the reporting module."""
    config_data = {
        'thresholds': {
            'temperature_alert': 40.0,
            'humidity_alert': 80.0,
            'air_quality_alert': 50.0
        },
        'correlation': {
            'fire_detection': [{
                'video_objects': ['fire', 'smoke'],
                'audio_sounds': ['Fire alarm', 'Fire'],
                'sensor_anomalies': ['temperature'],
                'confidence_threshold': 0.8
            }],
            'wildlife_detection': [{
                'video_objects': ['animal'],
                'audio_sounds': ['animal sound'],
                'confidence_threshold': 0.7
            }]
        }
    }
    with patch('src.integration.reporting.config', config_data):
        yield

def test_generate_consolidated_report(mock_config, sample_video_results, sample_audio_results, sample_sensor_anomalies):
    """Test the generation of a consolidated report."""
    report = generate_consolidated_report(sample_video_results, sample_audio_results, sample_sensor_anomalies)
    assert "Video Analysis Summary" in report
    assert "Audio Analysis Summary" in report
    assert "Sensor Data Anomaly Summary" in report
    assert "CONSOLIDATED ALERTS" in report
    assert "[HIGH PRIORITY] Potential Fire Risk" in report

def test_correlate_multimodal_data(mock_config, sample_video_results, sample_audio_results, sample_sensor_anomalies):
    """Test the cross-modal data correlation."""
    alerts = correlate_multimodal_data(sample_video_results, sample_audio_results, sample_sensor_anomalies)
    assert isinstance(alerts, list)
    assert len(alerts) > 0
    assert "[HIGH CONFIDENCE CORRELATED ALERT]" in alerts[0]

def test_correlate_multimodal_data_no_match(mock_config, sample_audio_results, sample_sensor_anomalies):
    """Test correlation with no video match."""
    alerts = correlate_multimodal_data({}, sample_audio_results, sample_sensor_anomalies)
    assert len(alerts) == 0

def test_correlate_multimodal_data_low_confidence(mock_config, sample_audio_results, sample_sensor_anomalies):
    """Test correlation with low confidence video results."""
    low_conf_video = {'fire': 0.5, 'smoke': 0.4}
    alerts = correlate_multimodal_data(low_conf_video, sample_audio_results, sample_sensor_anomalies)
    assert len(alerts) == 0

def test_correlate_wildlife_detection(mock_config):
    """Test wildlife detection correlation."""
    video_results = {'animal': 0.8}
    audio_results = {'animal sound': 0.75}
    alerts = correlate_multimodal_data(video_results, audio_results, None)
    assert len(alerts) > 0
    assert "[CORRELATED ALERT] Wildlife activity detected" in alerts[0]