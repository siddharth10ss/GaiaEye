import cv2
from ultralytics import YOLO
import os

def analyze_video(video_path):
    """
    Analyzes a video to detect objects using YOLOv8 and returns a summary.

    Args:
        video_path (str): The path to the video file.

    Returns:
        dict: A dictionary of detected objects and their highest confidence scores.
    """
    # Load the YOLOv8 model, supressing verbose output
    model = YOLO('yolov8n.pt', verbose=False)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return {}

    detected_objects = {}

    # Loop through the video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Process detected objects
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])

                # Store the highest confidence for each object class
                if class_name not in detected_objects or confidence > detected_objects[class_name]:
                    detected_objects[class_name] = confidence

    # Release the video capture object
    cap.release()

    return detected_objects

if __name__ == '__main__':
    # This part is for direct testing of the script.
    # It assumes the script is run from the root of the project.
    test_video_path = 'data/video/test_video.mp4'
    if not os.path.exists(test_video_path):
        print(f"Error: Test video not found at {test_video_path}")
    else:
        video_results = analyze_video(test_video_path)
        print("--- Video Analysis Results ---")
        if video_results:
            for obj, conf in video_results.items():
                print(f"- Detected: {obj} with confidence {conf:.2f}")
        else:
            print("No objects detected.")