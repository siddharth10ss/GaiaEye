# GaiaEye: AI-Powered Environmental Awareness System

GaiaEye is a prototype AI system that demonstrates multimodal environmental monitoring by processing video, audio, and simulated sensor data to detect environmental changes, recognize wildlife sounds, and predict potential hazards.

This prototype aims to showcase AI model integration across multiple modalities and the concept of correlating environmental data, suitable for academic demonstration, portfolio, and small-scale experiments.

## Features

*   **Modular Analysis Pipelines:** The system is built on a modular architecture with distinct, class-based pipelines for video, audio, and sensor analysis, making it efficient and extensible.
*   **Video Analysis:** Detects environmental objects using a pre-trained YOLOv8 model.
*   **Audio Analysis:** Classifies sounds using the pre-trained YAMNet model from TensorFlow Hub.
*   **Sensor Anomaly Detection:** Identifies anomalies in simulated sensor data using an Isolation Forest model.
*   **Cross-Modal Correlation:** Generates high-confidence alerts by correlating findings from different data sources (e.g., detecting "fire" in both video and audio).
*   **Centralized Configuration:** All system parameters, model paths, and thresholds are managed in a single `config.yaml` file for easy tuning.

## Project Structure

The project follows a modular structure to separate concerns and improve maintainability.

```
/
├── data/                 # Sample video, audio, and sensor data
│   ├── video/
│   ├── audio/
│   └── sensor/
├── src/                  # Main source code
│   ├── video_analysis/   # VideoAnalysisPipeline class
│   ├── audio_analysis/   # AudioAnalysisPipeline class
│   ├── sensor_analysis/  # SensorAnalysisPipeline class
│   ├── integration/      # Reporting and correlation logic
│   └── config.py         # Centralized configuration loader
├── tests/                # Unit and integration tests
├── main.py               # Main script to run the CLI application
├── config.yaml           # Configuration file for the system
├── requirements.txt      # Python dependencies for running the app
└── requirements-dev.txt  # Additional dependencies for testing and development
```

## Setup

Follow these steps to set up the project environment.

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd gaia-eye
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the dependencies:**
    *   To run the application:
        ```bash
        pip install -r requirements.txt
        ```
    *   To run tests and contribute to development, install the development dependencies as well:
        ```bash
        pip install -r requirements-dev.txt
        ```

## Usage

The application is run from the command line using `main.py`. You can specify paths to the video, audio, and sensor data files you wish to analyze. The sample files in the `data/` directory can be used for a quick test run.

**Run a full analysis using all sample data:**
```bash
python3 main.py --video data/video/test_video.mp4 --audio data/audio/white_noise.wav --sensor data/sensor/sensor_data.csv
```

**Run only a single analysis module:**
*   **Video only:**
    ```bash
    python3 main.py --video data/video/test_video.mp4
    ```
*   **Audio only:**
    ```bash
    python3 main.py --audio data/audio/sine_wave_1000hz.wav
    ```

The system will print a consolidated report to the console, including any alerts generated from the analysis.

## Testing

The project includes a suite of unit tests to ensure the correctness of each module.

1.  **Install development dependencies:**
    ```bash
    pip install -r requirements-dev.txt
    ```

2.  **Run the tests:**
    ```bash
    python3 -m pytest
    ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.