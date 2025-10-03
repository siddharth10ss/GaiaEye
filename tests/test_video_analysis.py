import pytest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import patch, MagicMock

# Add the src directory to the path


from src.video_analysis.analysis import analyze_video

def test_analyze_video_with_valid_input():
    """Test video analysis with valid input."""
    # Mock the YOLO model and cv2.VideoCapture
    with patch('src.video_analysis.analysis.YOLO') as mock_yolo, \
         patch('src.video_analysis.analysis.cv2.VideoCapture') as mock_video_capture:
        
        # Mock the video capture object
        mock_cap = MagicMock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),  # First frame
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),  # Second frame
            (False, None)  # End of video
        ]
        # Mock get to return frame count and fps
        mock_cap.get.side_effect = lambda prop: {
            7: 3,  # CV_CAP_PROP_FRAME_COUNT
            5: 30.0  # CV_CAP_PROP_FPS
        }.get(prop, 0)
        
        # Mock the YOLO model instance
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        # Mock the model attributes and inference results
        mock_model.names = {0: 'person'}

        # Mock the results
        mock_boxes = MagicMock()
        mock_boxes.cls = [np.array([0])]  # class id
        mock_boxes.conf = [np.array([0.9])]  # confidence

        mock_result = MagicMock()
        mock_result.boxes = [mock_boxes]

        # Mock the model inference call
        mock_model.return_value = [mock_result]
        
        # Test the function with sample rate of 1 to ensure processing
        config = {'models': {'yolo': 'yolov8n.pt'}, 'processing': {'video_fps_sample': 1}}
        results = analyze_video('test_video.mp4', config)
        
        # Assertions
        assert isinstance(results, dict)
        assert 'person' in results
        assert results['person'] == 0.9

def test_analyze_video_with_invalid_file():
    """Test video analysis with invalid file path."""
    config = {'models': {'yolo': 'yolov8n.pt'}}
    results = analyze_video('nonexistent_video.mp4', config)
    assert isinstance(results, dict)
    assert len(results) == 0

def test_analyze_video_with_empty_config():
    """Test video analysis with empty config."""
    with patch('src.video_analysis.analysis.YOLO') as mock_yolo, \
         patch('src.video_analysis.analysis.cv2.VideoCapture') as mock_video_capture:
        
        # Mock the video capture object
        mock_cap = MagicMock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)  # Empty video
        
        # Mock the YOLO model
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model
        
        # Test the function
        results = analyze_video('test_video.mp4', {})
        
        # Assertions
        assert isinstance(results, dict)
