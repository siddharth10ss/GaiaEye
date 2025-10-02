import argparse
import os
import sys

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from video_analysis.analysis import analyze_video
from audio_analysis.analysis import analyze_audio
from sensor_analysis.analysis import analyze_sensor_data
from integration.reporting import generate_consolidated_report

def main():
    """
    Main function to run the GaiaEye prototype.
    """
    parser = argparse.ArgumentParser(description="GaiaEye: AI-Powered Environmental Awareness System")
    parser.add_argument('--video', type=str, help='Path to the video file to analyze.')
    parser.add_argument('--audio', type=str, help='Path to the audio file to analyze.')
    parser.add_argument('--sensor', type=str, help='Path to the sensor data CSV file to analyze.')

    args = parser.parse_args()

    video_results = {}
    audio_results = {}
    sensor_anomalies = None

    if args.video:
        if os.path.exists(args.video):
            video_results = analyze_video(args.video)
        else:
            print(f"Error: Video file not found at {args.video}")

    if args.audio:
        if os.path.exists(args.audio):
            audio_results = analyze_audio(args.audio)
        else:
            print(f"Error: Audio file not found at {args.audio}")

    if args.sensor:
        if os.path.exists(args.sensor):
            sensor_anomalies = analyze_sensor_data(args.sensor)
        else:
            print(f"Error: Sensor data file not found at {args.sensor}")

    # Generate and print the consolidated report
    report = generate_consolidated_report(video_results, audio_results, sensor_anomalies)
    print(report)

if __name__ == '__main__':
    main()