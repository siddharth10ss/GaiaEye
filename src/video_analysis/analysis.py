import cv2
from ultralytics import YOLO
import os
import logging
from typing import Dict, Any, Optional
from tqdm import tqdm

def analyze_video(video_path: str, config: Optional[Dict[Any, Any]] = None) -> Dict[str, float]:
    """
    Analyzes a video to detect objects using YOLOv8 and returns a summary.

    Args:
        video_path (str): The path to the video file.
        config (dict, optional): Configuration dictionary.

    Returns:
        dict: A dictionary of detected objects and their highest confidence scores.
    """
    if config is None:
        config = {}
    
    # Get configuration values with defaults
    model_path = config.get('models', {}).get('yolo', 'yolov8n.pt')
    sample_rate = config.get('processing', {}).get('video_fps_sample', 1)
    confidence_threshold = config.get('thresholds', {}).get('confidence_threshold', 0.5)
    
    try:
        # Load the YOLOv8 model, suppressing verbose output
        model = YOLO(model_path, verbose=False)
        logging.info(f"Loaded YOLO model from {model_path}")
    except Exception as e:
        logging.error(f"Failed to load YOLO model: {e}")
        return {}

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logging.error(f"Error: Could not open video {video_path}")
        return {}

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    logging.info(f"Processing video: {video_path}")
    logging.info(f"Total frames: {total_frames}, FPS: {fps}")
    
    detected_objects = {}
    frame_count = 0

    # Initialize progress bar
    pbar = tqdm(total=total_frames, desc="Processing video frames", unit="frames")

    # Loop through the video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame based on sampling rate
        if frame_count % sample_rate == 0:
            try:
                # Run YOLOv8 inference on the frame
                results = model(frame)

                # Process detected objects
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        class_id = int(box.cls[0].item())
                        class_name = model.names[class_id]
                        confidence = float(box.conf[0].item())
                        
                        # Only consider detections above confidence threshold
                        if confidence >= confidence_threshold:
                            # Store the highest confidence for each object class
                            if class_name not in detected_objects or confidence > detected_objects[class_name]:
                                detected_objects[class_name] = confidence
            except Exception as e:
                logging.warning(f"Error processing frame {frame_count}: {e}")
        
        frame_count += 1
        pbar.update(1)

    # Release the video capture object
    cap.release()
    pbar.close()
    
    logging.info(f"Video analysis complete. Detected {len(detected_objects)} object types.")
    
    return detected_objects

if __name__ == '__main__':
    # This part is for direct testing of the script.
    # It assumes the script is run from the root of the project.
    import yaml
    
    # Load config for testing
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except:
        config = {}
    
    test_video_path = 'data/video/test_video.mp4'
    if not os.path.exists(test_video_path):
        print(f"Error: Test video not found at {test_video_path}")
    else:
        video_results = analyze_video(test_video_path, config)
        print("--- Video Analysis Results ---")
        if video_results:
            for obj, conf in video_results.items():
                print(f"- Detected: {obj} with confidence {conf:.2f}")
        else:
            print("No objects detected.")
