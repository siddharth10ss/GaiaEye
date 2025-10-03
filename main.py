import argparse
import os
import sys
import logging
import yaml

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.video_analysis.analysis import analyze_video
from src.audio_analysis.analysis import analyze_audio
from src.sensor_analysis.analysis import analyze_sensor_data
from src.integration.reporting import generate_consolidated_report, correlate_multimodal_data

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logging.warning(f"Config file {config_path} not found. Using default settings.")
        return {}
    except yaml.YAMLError as e:
        logging.error(f"Error parsing config file: {e}")
        return {}

def setup_logging(config: dict) -> None:
    """Set up logging based on configuration."""
    log_config = config.get('logging', {})
    logging.basicConfig(
        level=getattr(logging, log_config.get('level', 'INFO')),
        format=log_config.get('format', '%(asctime)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(log_config.get('file', 'gaiaeye.log')),
            logging.StreamHandler()
        ]
    )

def validate_file_path(file_path: str, file_type: str) -> bool:
    """Validate if file exists and has correct extension."""
    if not os.path.exists(file_path):
        logging.error(f"{file_type.capitalize()} file not found at {file_path}")
        return False
    
    # Check file extension based on type
    if file_type == "video":
        valid_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    elif file_type == "audio":
        valid_extensions = ['.wav', '.mp3', '.flac']
    elif file_type == "sensor":
        valid_extensions = ['.csv']
    else:
        return True  # No validation for unknown types
    
    if not any(file_path.endswith(ext) for ext in valid_extensions):
        logging.warning(f"{file_type.capitalize()} file has unexpected extension: {file_path}")
    
    return True

def main():
    """
    Main function to run the GaiaEye prototype.
    """
    parser = argparse.ArgumentParser(description="GaiaEye: AI-Powered Environmental Awareness System")
    parser.add_argument('--video', type=str, help='Path to the video file to analyze.')
    parser.add_argument('--audio', type=str, help='Path to the audio file to analyze.')
    parser.add_argument('--sensor', type=str, help='Path to the sensor data CSV file to analyze.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file.')

    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up logging
    setup_logging(config)
    logging.info("GaiaEye system starting...")
    
    video_results = {}
    audio_results = {}
    sensor_anomalies = None

    try:
        if args.video:
            if validate_file_path(args.video, "video"):
                logging.info(f"Analyzing video: {args.video}")
                video_results = analyze_video(args.video, config)
            else:
                logging.error(f"Video file validation failed: {args.video}")

        if args.audio:
            if validate_file_path(args.audio, "audio"):
                logging.info(f"Analyzing audio: {args.audio}")
                audio_results = analyze_audio(args.audio, config)
            else:
                logging.error(f"Audio file validation failed: {args.audio}")

        if args.sensor:
            if validate_file_path(args.sensor, "sensor"):
                logging.info(f"Analyzing sensor data: {args.sensor}")
                sensor_anomalies = analyze_sensor_data(args.sensor, config)
            else:
                logging.error(f"Sensor data file validation failed: {args.sensor}")

        # Generate and print the consolidated report with cross-modal correlation
        report = generate_consolidated_report(video_results, audio_results, sensor_anomalies, config)
        correlated_alerts = correlate_multimodal_data(video_results, audio_results, sensor_anomalies, config)
        
        print(report)
        if correlated_alerts:
            print("\n--- CROSS-MODAL CORRELATION ALERTS ---")
            for alert in correlated_alerts:
                print(f"- {alert}")
        
        logging.info("GaiaEye analysis completed successfully.")
        
    except Exception as e:
        logging.error(f"An error occurred during analysis: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
