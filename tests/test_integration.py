import pytest
import pandas as pd

from src.integration.reporting import generate_consolidated_report, correlate_multimodal_data

@pytest.fixture
def sample_video_results():
    return {'fire': 0.9, 'smoke': 0.8}

@pytest.fixture
def sample_audio_results():
    return {'Fire alarm': 0.85, 'Fire': 0.7}

@pytest.fixture
def sample_sensor_anomalies():
    data = {
        'timestamp': pd.to_datetime(['2023-01-01 12:02:00']),
        'temperature': [50.0],  # Anomaly
        'humidity': [55.0],
        'air_quality': [80.0]  # Anomaly
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_config():
    return {
        'thresholds': {
            'temperature_alert': 40.0,
            'humidity_alert': 80.0,
            'air_quality_alert': 50.0
        },
        'correlation': {
            'fire_detection': [
                {
                    'video_objects': ['fire', 'smoke'],
                    'audio_sounds': ['Fire alarm', 'Fire'],
                    'sensor_anomalies': ['temperature'],
                    'confidence_threshold': 0.8
                }
            ]
        }
    }

def test_generate_consolidated_report(sample_video_results, sample_audio_results, sample_sensor_anomalies, sample_config):
    """Test the generation of a consolidated report."""
    report = generate_consolidated_report(sample_video_results, sample_audio_results, sample_sensor_anomalies, sample_config)

    assert "Video Analysis Summary" in report
    assert "Audio Analysis Summary" in report
    assert "Sensor Data Anomaly Summary" in report
    assert "CONSOLIDATED ALERTS" in report
    assert "[HIGH PRIORITY] Potential Fire Risk" in report

def test_correlate_multimodal_data(sample_video_results, sample_audio_results, sample_sensor_anomalies, sample_config):
    """Test the cross-modal data correlation."""
    alerts = correlate_multimodal_data(sample_video_results, sample_audio_results, sample_sensor_anomalies, sample_config)

    assert isinstance(alerts, list)
    assert len(alerts) > 0
    assert "[HIGH CONFIDENCE CORRELATED ALERT]" in alerts[0]
