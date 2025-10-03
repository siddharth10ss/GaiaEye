# GaiaEye: AI-Powered Environmental Awareness System

GaiaEye is a prototype AI system that demonstrates multimodal environmental monitoring by processing video, audio, and simulated sensor data to detect environmental changes, recognize wildlife sounds, and predict potential hazards. The system generates alerts and summaries in real-time.

This prototype aims to showcase AI model integration across multiple modalities and the concept of correlating environmental data, suitable for academic demonstration, portfolio, and small-scale experiments.

## Features

*   **Video Analysis:** Detects environmental objects and changes using a pre-trained YOLOv8 model.
*   **Audio Analysis:** Classifies natural sounds using the pre-trained YAMNet audio classification model.
*   **Sensor Correlation:** Predicts environmental hazards by detecting anomalies in simulated sensor data using an Isolation Forest model.
*   **Alerts & Summaries:** Generates real-time textual alerts and a consolidated summary of the findings from all analysis modules.

## Project Structure

```
/
├── data/             # Directory for sample video, audio, and CSV files
│   ├── video/
│   ├── audio/
│   └── sensor/
├── src/              # Main source code
│   ├── video_analysis/
│   ├── audio_analysis/
│   ├── sensor_analysis/
│   └── integration/
├── main.py           # Main script to run the CLI application
└── requirements.txt  # Python dependencies
```

## Setup

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
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The application is run from the command line using `main.py`. You can specify the paths to the video, audio, and sensor data files to be analyzed.

**Example:**

```bash
python3 main.py --video data/video/test_video.mp4 --audio data/audio/white_noise.wav --sensor data/sensor/sensor_data.csv
```

This will run all three analysis modules and print a consolidated report to the console, including any alerts generated from the analysis. You can also run each module individually by providing only the relevant argument (e.g., `--video <path_to_video>`).

## Testing

The project includes a suite of unit tests to ensure the correctness of each module. The tests cover the video, audio, sensor, and integration modules.

To run the tests, use the following command:

```bash
pytest
```

## Future Improvements

*   **Real-time Data Streaming:** Instead of processing static files, the system could be enhanced to handle real-time data streams from cameras, microphones, and sensors.
*   **Advanced Anomaly Detection:** The sensor data analysis could be improved with more sophisticated anomaly detection models, such as LSTMs or other deep learning models, to better capture temporal dependencies.
*   **Scalability:** For larger-scale deployments, the system could be integrated with a message queue (like RabbitMQ or Kafka) and a distributed processing framework (like Spark or Dask) to handle a high volume of data from multiple sources.
*   **Dashboard and Visualization:** A web-based dashboard could be developed to visualize the data, alerts, and analysis results in a more user-friendly way.
*   **Model Retraining and Fine-tuning:** The AI models could be retrained or fine-tuned on custom datasets to improve their accuracy for specific environments or use cases.
*   **Expanded Sound Library:** The audio analysis could be expanded to recognize a wider range of sounds, including specific types of wildlife, machinery, or other environmental indicators.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.