import pytest
import numpy as np
import os
from unittest.mock import patch, MagicMock

from src.video_analysis.analysis import VideoAnalysisPipeline

@pytest.fixture
def mock_config():
    """Fixture to mock the centralized config."""
    with patch('src.video_analysis.analysis.config', {
        'video_analysis': {
            'model_path': 'yolov8n.pt',
            'sample_rate': 1,
            'confidence_threshold': 0.5
        }
    }) as mock_cfg:
        yield mock_cfg

def test_analyze_video_with_valid_input(mock_config):
    """Test video analysis with valid input using the new pipeline."""
    with patch('src.video_analysis.analysis.YOLO') as mock_yolo, \
         patch('src.video_analysis.analysis.cv2.VideoCapture') as mock_video_capture:
        
        # Mock the YOLO model instance created in the pipeline's __init__
        mock_model_instance = MagicMock()
        mock_model_instance.names = {0: 'person'}
        mock_yolo.return_value = mock_model_instance
        
        # Mock the inference results to be iterable
        mock_box = MagicMock()
        # ultralytics results have .cls and .conf as tensors. .item() extracts the value.
        mock_box.cls = [MagicMock(item=MagicMock(return_value=0))]
        mock_box.conf = [MagicMock(item=MagicMock(return_value=0.9))]

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]  # boxes is an iterable of box objects
        mock_model_instance.return_value = [mock_result]

        # Mock the video capture object
        mock_cap_instance = MagicMock()
        mock_cap_instance.isOpened.return_value = True
        mock_cap_instance.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (False, None)
        ]
        mock_cap_instance.get.return_value = 1 # Mock frame count
        mock_video_capture.return_value = mock_cap_instance
        
        # Instantiate the pipeline, which will use the mocked YOLO model
        video_pipeline = VideoAnalysisPipeline()

        # Run the analysis
        results = video_pipeline.analyze('dummy_video.mp4')
        
        # Assertions
        assert video_pipeline.model is not None
        assert 'person' in results
        assert results['person'] == 0.9
        mock_yolo.assert_called_with('yolov8n.pt', verbose=False)

def test_analyze_video_with_invalid_file(mock_config):
    """Test video analysis with an invalid file path."""
    with patch('src.video_analysis.analysis.YOLO'): # Still need to patch model loading
        video_pipeline = VideoAnalysisPipeline()
        results = video_pipeline.analyze('nonexistent_video.mp4')
        assert results == {}

def test_video_pipeline_model_loading_failure():
    """Test that analysis is skipped if the model fails to load."""
    with patch('src.video_analysis.analysis.YOLO', side_effect=Exception("Model load error")):
        video_pipeline = VideoAnalysisPipeline()
        assert video_pipeline.model is None
        results = video_pipeline.analyze('dummy_video.mp4')
        assert results == {}