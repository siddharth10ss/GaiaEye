import pandas as pd

def generate_consolidated_report(video_results, audio_results, sensor_anomalies):
    """
    Generates a consolidated report from the analysis of video, audio, and sensor data.

    Args:
        video_results (dict): A dictionary of detected objects from video analysis.
        audio_results (dict): A dictionary of detected sounds from audio analysis.
        sensor_anomalies (pd.DataFrame or None): A DataFrame of detected sensor anomalies.
    """
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
    # Simple alert rule: High temperature anomaly
    if sensor_anomalies is not None and not sensor_anomalies.empty:
        if sensor_anomalies['temperature'].max() > 40:
            alerts.append("[HIGH PRIORITY] Potential Fire Risk: Unusually high temperature detected.")
        alerts.append(f"[INFO] {len(sensor_anomalies)} sensor anomalies detected.")

    if not alerts:
        report += "All systems normal. No alerts to report.\n"
    else:
        for alert in alerts:
            report += f"- {alert}\n"

    return report