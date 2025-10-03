import argparse
import os
import sys
import logging

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.config import config
from src.video_analysis.analysis import VideoAnalysisPipeline
from src.audio_analysis.analysis import AudioAnalysisPipeline
from src.sensor_analysis.analysis import SensorAnalysisPipeline
from src.integration.reporting import generate_consolidated_report, correlate_multimodal_data

def setup_logging() -> None:
    """Set up logging based on the centralized configuration."""
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
    
    valid_extensions = {
        "video": ['.mp4', '.avi', '.mov', '.mkv'],
        "audio": ['.wav', '.mp3', '.flac'],
        "sensor": ['.csv']
    }.get(file_type, [])
    
    if valid_extensions and not any(file_path.lower().endswith(ext) for ext in valid_extensions):
        logging.warning(f"{file_type.capitalize()} file has unexpected extension: {file_path}")
    
    return True

def main():
    """
    Main function to run the GaiaEye prototype.
    """
    setup_logging()
    logging.info("GaiaEye system starting... Initializing analysis pipelines.")

    # Initialize pipelines
    video_pipeline = VideoAnalysisPipeline()
    audio_pipeline = AudioAnalysisPipeline()
    sensor_pipeline = SensorAnalysisPipeline()

    logging.info("Pipelines initialized.")

    parser = argparse.ArgumentParser(description="GaiaEye: AI-Powered Environmental Awareness System")
    parser.add_argument('--video', type=str, help='Path to the video file to analyze.')
    parser.add_argument('--audio', type=str, help='Path to the audio file to analyze.')
    parser.add_argument('--sensor', type=str, help='Path to the sensor data CSV file to analyze.')

    args = parser.parse_args()
    
    video_results = {}
    audio_results = {}
    sensor_anomalies = None

    try:
        if args.video:
            if video_pipeline.model and validate_file_path(args.video, "video"):
                logging.info(f"Analyzing video: {args.video}")
                video_results = video_pipeline.analyze(args.video)
            else:
                logging.error(f"Video analysis skipped for {args.video} due to validation or model loading failure.")

        if args.audio:
            if audio_pipeline.model and validate_file_path(args.audio, "audio"):
                logging.info(f"Analyzing audio: {args.audio}")
                audio_results = audio_pipeline.analyze(args.audio)
            else:
                logging.error(f"Audio analysis skipped for {args.audio} due to validation or model loading failure.")

        if args.sensor:
            if validate_file_path(args.sensor, "sensor"):
                logging.info(f"Analyzing sensor data: {args.sensor}")
                sensor_anomalies = sensor_pipeline.analyze(args.sensor)
            else:
                logging.error(f"Sensor data analysis skipped due to file validation failure: {args.sensor}")

        # Generate and print the consolidated report
        report = generate_consolidated_report(video_results, audio_results, sensor_anomalies)
        correlated_alerts = correlate_multimodal_data(video_results, audio_results, sensor_anomalies)
        
        print(report)
        if correlated_alerts:
            print("\n--- CROSS-MODAL CORRELATION ALERTS ---")
            for alert in correlated_alerts:
                print(f"- {alert}")
        
        logging.info("GaiaEye analysis completed successfully.")
        
    except Exception as e:
        logging.error(f"An unexpected error occurred during the main execution: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()