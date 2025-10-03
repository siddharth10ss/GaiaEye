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
def no_sensor_anomalies():
    return pd.DataFrame()

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
            ],
            'wildlife_detection': [
                {
                    'video_objects': ['animal'],
                    'audio_sounds': ['animal sound'],
                    'confidence_threshold': 0.7
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

def test_generate_consolidated_report_no_video(sample_audio_results, sample_sensor_anomalies, sample_config):
    """Test report generation with no video results."""
    report = generate_consolidated_report({}, sample_audio_results, sample_sensor_anomalies, sample_config)
    assert "No significant objects detected." in report

def test_generate_consolidated_report_no_audio(sample_video_results, sample_sensor_anomalies, sample_config):
    """Test report generation with no audio results."""
    report = generate_consolidated_report(sample_video_results, {}, sample_sensor_anomalies, sample_config)
    assert "No significant sounds detected." in report

def test_generate_consolidated_report_no_sensor_anomalies(sample_video_results, sample_audio_results, no_sensor_anomalies, sample_config):
    """Test report generation with no sensor anomalies."""
    report = generate_consolidated_report(sample_video_results, sample_audio_results, no_sensor_anomalies, sample_config)
    assert "No anomalies detected in sensor data." in report
    assert "All systems normal. No alerts to report." in report

def test_correlate_multimodal_data(sample_video_results, sample_audio_results, sample_sensor_anomalies, sample_config):
    """Test the cross-modal data correlation."""
    alerts = correlate_multimodal_data(sample_video_results, sample_audio_results, sample_sensor_anomalies, sample_config)

    assert isinstance(alerts, list)
    assert len(alerts) > 0
    assert "[HIGH CONFIDENCE CORRELATED ALERT]" in alerts[0]

def test_correlate_multimodal_data_no_video_match(sample_audio_results, sample_sensor_anomalies, sample_config):
    """Test correlation with no video match."""
    alerts = correlate_multimodal_data({}, sample_audio_results, sample_sensor_anomalies, sample_config)
    assert len(alerts) == 0

def test_correlate_multimodal_data_no_audio_match(sample_video_results, sample_sensor_anomalies, sample_config):
    """Test correlation with no audio match."""
    alerts = correlate_multimodal_data(sample_video_results, {}, sample_sensor_anomalies, sample_config)
    assert len(alerts) == 0

def test_correlate_multimodal_data_no_sensor_match(sample_video_results, sample_audio_results, no_sensor_anomalies, sample_config):
    """Test correlation with no sensor match."""
    alerts = correlate_multimodal_data(sample_video_results, sample_audio_results, no_sensor_anomalies, sample_config)
    assert len(alerts) == 0

def test_correlate_multimodal_data_low_confidence(sample_video_results, sample_audio_results, sample_sensor_anomalies, sample_config):
    """Test correlation with low confidence."""
    low_conf_video = {'fire': 0.5, 'smoke': 0.4}
    alerts = correlate_multimodal_data(low_conf_video, sample_audio_results, sample_sensor_anomalies, sample_config)
    assert len(alerts) == 0

def test_correlate_wildlife_detection(sample_config):
    """Test wildlife detection correlation."""
    video_results = {'animal': 0.8}
    audio_results = {'animal sound': 0.75}
    alerts = correlate_multimodal_data(video_results, audio_results, None, sample_config)
    assert len(alerts) > 0
    assert "[CORRELATED ALERT] Wildlife activity detected" in alerts[0]