import pandas as pd
import logging
from typing import Dict, Any, Optional, List

def generate_consolidated_report(
    video_results: Dict[str, float], 
    audio_results: Dict[str, float], 
    sensor_anomalies: Optional[pd.DataFrame],
    config: Optional[Dict[Any, Any]] = None
) -> str:
    """
    Generates a consolidated report from the analysis of video, audio, and sensor data.

    Args:
        video_results (dict): A dictionary of detected objects from video analysis.
        audio_results (dict): A dictionary of detected sounds from audio analysis.
        sensor_anomalies (pd.DataFrame or None): A DataFrame of detected sensor anomalies.
        config (dict, optional): Configuration dictionary.

    Returns:
        str: Formatted report string.
    """
    if config is None:
        config = {}
    
    # Get threshold values from config
    temp_threshold = config.get('thresholds', {}).get('temperature_alert', 40.0)
    humidity_threshold = config.get('thresholds', {}).get('humidity_alert', 80.0)
    air_quality_threshold = config.get('thresholds', {}).get('air_quality_alert', 50.0)
    
    report = "--- GaiaEye Consolidated Environmental Report ---\n\n"

    # --- Video Analysis Summary ---
    report += "--- Video Analysis Summary ---\n"
    if video_results:
        report += "Detected objects:\n"
        for obj, conf in video_results.items():
            report += f"- {obj} (Confidence: {conf:.2f})\n"
    else:
        report += "No significant objects detected.\n"

    # --- Audio Analysis Summary ---
    report += "\n--- Audio Analysis Summary ---\n"
    if audio_results:
        report += "Detected sounds:\n"
        for sound, score in audio_results.items():
            report += f"- {sound} (Score: {score:.3f})\n"
    else:
        report += "No significant sounds detected.\n"

    # --- Sensor Data Summary ---
    report += "\n--- Sensor Data Anomaly Summary ---\n"
    if sensor_anomalies is not None and not sensor_anomalies.empty:
        report += "Detected sensor anomalies:\n"
        for index, row in sensor_anomalies.iterrows():
            report += f"- Timestamp: {row['timestamp']}, Temp: {row['temperature']}, Humidity: {row['humidity']}, Air Quality: {row['air_quality']}\n"
    else:
        report += "No anomalies detected in sensor data.\n"

    # --- Consolidated Alerts ---
    report += "\n--- CONSOLIDATED ALERTS ---\n"
    alerts = []
    
    # Sensor-based alerts
    if sensor_anomalies is not None and not sensor_anomalies.empty:
        max_temp = sensor_anomalies['temperature'].max()
        max_humidity = sensor_anomalies['humidity'].max()
        max_air_quality = sensor_anomalies['air_quality'].max()
        
        if max_temp > temp_threshold:
            alerts.append(f"[HIGH PRIORITY] Potential Fire Risk: Unusually high temperature detected ({max_temp:.1f}°C).")
        
        if max_humidity > humidity_threshold:
            alerts.append(f"[MEDIUM PRIORITY] High Humidity: Very humid conditions detected ({max_humidity:.1f}%).")
            
        if max_air_quality > air_quality_threshold:
            alerts.append(f"[MEDIUM PRIORITY] Poor Air Quality: Air quality index is high ({max_air_quality:.1f}).")
            
        alerts.append(f"[INFO] {len(sensor_anomalies)} sensor anomalies detected.")

    if not alerts:
        report += "All systems normal. No alerts to report.\n"
    else:
        for alert in alerts:
            report += f"- {alert}\n"

    return report

def correlate_multimodal_data(
    video_results: Dict[str, float], 
    audio_results: Dict[str, float], 
    sensor_anomalies: Optional[pd.DataFrame],
    config: Optional[Dict[Any, Any]] = None
) -> List[str]:
    """
    Performs cross-modal correlation to identify complex environmental events.
    
    Args:
        video_results (dict): Detected objects from video analysis.
        audio_results (dict): Detected sounds from audio analysis.
        sensor_anomalies (pd.DataFrame or None): Sensor anomalies.
        config (dict, optional): Configuration dictionary.
        
    Returns:
        list: List of correlated alerts.
    """
    if config is None:
        config = {}
    
    correlated_alerts = []
    
    # Get correlation rules from config
    correlation_rules = config.get('correlation', {})
    
    # Fire detection correlation
    fire_rules = correlation_rules.get('fire_detection', [])
    for rule in fire_rules:
        video_objects = rule.get('video_objects', [])
        audio_sounds = rule.get('audio_sounds', [])
        sensor_anomalies_list = rule.get('sensor_anomalies', [])
        confidence_threshold = rule.get('confidence_threshold', 0.8)
        
        # Check if we have matches in all modalities
        video_match = any(obj in video_objects for obj in video_results.keys())
        audio_match = any(sound in audio_sounds for sound in audio_results.keys())
        
        sensor_match = False
        if sensor_anomalies is not None and not sensor_anomalies.empty:
            sensor_match = any(
                anomaly_type in sensor_anomalies.columns and 
                sensor_anomalies[anomaly_type].max() > config.get('thresholds', {}).get(f'{anomaly_type}_alert', 0)
                for anomaly_type in sensor_anomalies_list
            )
        
        # If we have matches in all modalities, generate a correlated alert
        if video_match and audio_match and sensor_match:
            # Calculate confidence based on individual scores
            video_conf = max([video_results.get(obj, 0) for obj in video_objects if obj in video_results], default=0)
            audio_conf = max([audio_results.get(sound, 0) for sound in audio_sounds if sound in audio_results], default=0)
            
            # Simple weighted average for correlation confidence
            correlation_confidence = (video_conf + audio_conf) / 2
            
            if correlation_confidence >= confidence_threshold:
                correlated_alerts.append(
                    f"[HIGH CONFIDENCE CORRELATED ALERT] Fire event detected with {correlation_confidence:.2f} confidence: "
                    f"Visual ({', '.join([obj for obj in video_objects if obj in video_results])}), "
                    f"Audio ({', '.join([sound for sound in audio_sounds if sound in audio_results])}), "
                    f"Sensor anomalies confirmed."
                )
    
    # Wildlife detection correlation
    wildlife_rules = correlation_rules.get('wildlife_detection', [])
    for rule in wildlife_rules:
        video_objects = rule.get('video_objects', [])
        audio_sounds = rule.get('audio_sounds', [])
        confidence_threshold = rule.get('confidence_threshold', 0.7)
        
        # Check if we have matches in both video and audio
        video_match = any(obj in video_objects for obj in video_results.keys())
        audio_match = any(sound in audio_sounds for sound in audio_results.keys())
        
        # If we have matches in both modalities, generate a correlated alert
        if video_match and audio_match:
            # Calculate confidence based on individual scores
            video_conf = max([video_results.get(obj, 0) for obj in video_objects if obj in video_results], default=0)
            audio_conf = max([audio_results.get(sound, 0) for sound in audio_sounds if sound in audio_results], default=0)
            
            # Simple average for correlation confidence
            correlation_confidence = (video_conf + audio_conf) / 2
            
            if correlation_confidence >= confidence_threshold:
                correlated_alerts.append(
                    f"[CORRELATED ALERT] Wildlife activity detected with {correlation_confidence:.2f} confidence: "
                    f"Visual ({', '.join([obj for obj in video_objects if obj in video_results])}), "
                    f"Audio ({', '.join([sound for sound in audio_sounds if sound in audio_results])})"
                )
    
    logging.info(f"Cross-modal correlation completed. Generated {len(correlated_alerts)} correlated alerts.")
    return correlated_alerts

if __name__ == '__main__':
    # Example usage for testing
    video_results = {"person": 0.9, "dog": 0.7}
    audio_results = {"Speech": 0.8, "Dog": 0.6}
    sensor_anomalies = pd.DataFrame({
        'timestamp': ['2025-10-02T12:05:00Z'],
        'temperature': [45.0],
        'humidity': [75.0],
        'air_quality': [50.0],
        'anomaly': [-1]
    })
    
    # Load config for testing
    import yaml
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except:
        config = {}
    
    report = generate_consolidated_report(video_results, audio_results, sensor_anomalies, config)
    correlated_alerts = correlate_multimodal_data(video_results, audio_results, sensor_anomalies, config)
    
    print(report)
    if correlated_alerts:
        print("\n--- CROSS-MODAL CORRELATION ALERTS ---")
        for alert in correlated_alerts:
            print(f"- {alert}")
