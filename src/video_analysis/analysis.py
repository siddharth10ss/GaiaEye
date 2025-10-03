import cv2
from ultralytics import YOLO
import os
import logging
from typing import Dict, Any
from tqdm import tqdm
from src.config import config

class VideoAnalysisPipeline:
    """
    A pipeline for analyzing video to detect objects using YOLOv8.
    The model is loaded once during initialization for efficiency.
    """
    def __init__(self):
        """
        Initializes the video analysis pipeline by loading the model and configuration.
        """
        video_config = config.get('video_analysis', {})
        model_path = video_config.get('model_path', 'yolov8n.pt')
        self.sample_rate = video_config.get('sample_rate', 1)
        self.confidence_threshold = video_config.get('confidence_threshold', 0.5)

        try:
            self.model = YOLO(model_path, verbose=False)
            logging.info(f"Loaded YOLO model from {model_path}")
        except Exception as e:
            logging.error(f"Failed to load YOLO model: {e}")
            self.model = None

    def analyze(self, video_path: str) -> Dict[str, float]:
        """
        Analyzes a video to detect objects using the pre-loaded YOLOv8 model.

        Args:
            video_path (str): The path to the video file.

        Returns:
            dict: A dictionary of detected objects and their highest confidence scores.
        """
        if not self.model:
            logging.error("YOLO model is not loaded. Cannot analyze video.")
            return {}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Error: Could not open video {video_path}")
            return {}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logging.info(f"Processing video: {video_path} ({total_frames} frames)")

        detected_objects = {}
        frame_count = 0
        pbar = tqdm(total=total_frames, desc="Processing video frames", unit="frames")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % self.sample_rate == 0:
                try:
                    results = self.model(frame)
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            class_id = int(box.cls[0].item())
                            class_name = self.model.names[class_id]
                            confidence = float(box.conf[0].item())

                            if confidence >= self.confidence_threshold:
                                if class_name not in detected_objects or confidence > detected_objects[class_name]:
                                    detected_objects[class_name] = confidence
                except Exception as e:
                    logging.warning(f"Error processing frame {frame_count}: {e}")

            frame_count += 1
            pbar.update(1)

        cap.release()
        pbar.close()
        
        logging.info(f"Video analysis complete. Detected {len(detected_objects)} object types.")
        return detected_objects

if __name__ == '__main__':
    # This part is for direct testing of the script.
    # It assumes the script is run from the root of the project.
    
    test_video_path = 'data/video/test_video.mp4'

    if not os.path.exists(test_video_path):
        print(f"Error: Test video not found at {test_video_path}")
    else:
        # Instantiate the pipeline and run the analysis
        video_pipeline = VideoAnalysisPipeline()
        if video_pipeline.model:
            video_results = video_pipeline.analyze(test_video_path)
            print("--- Video Analysis Results ---")
            if video_results:
                for obj, conf in video_results.items():
                    print(f"- Detected: {obj} with confidence {conf:.2f}")
            else:
                print("No objects detected.")
        else:
            print("Could not run video analysis due to model loading failure.")