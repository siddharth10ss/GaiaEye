import pytest
from unittest.mock import patch, MagicMock, mock_open
import numpy as np

# This import is necessary to test the pipeline, which depends on tensorflow
try:
    import tensorflow as tf
except ImportError:
    tf = None

from src.audio_analysis.analysis import AudioAnalysisPipeline

@pytest.fixture
def mock_config():
    """Fixture to mock the centralized config for audio analysis."""
    with patch('src.audio_analysis.analysis.config', {
        'audio_analysis': {
            'yamnet_url': 'https://tfhub.dev/google/yamnet/1',
            'audio_score_threshold': 0.3
        }
    }) as mock_cfg:
        yield mock_cfg

@pytest.mark.skipif(tf is None, reason="TensorFlow is not installed")
@patch('src.audio_analysis.analysis.hub.load')
def test_analyze_audio_with_valid_input(mock_hub_load, mock_config):
    """Test audio analysis with valid input using the pipeline."""
    # Mock the model and its methods called in __init__
    mock_model = MagicMock()
    # Create a mock for the numpy method
    mock_numpy = MagicMock()
    mock_numpy.return_value = 'class_map.csv'
    # Create a mock for the class_map_path method
    mock_class_map_path = MagicMock()
    mock_class_map_path.return_value.numpy = mock_numpy
    mock_model.class_map_path = mock_class_map_path
    mock_hub_load.return_value = mock_model

    # Mock the CSV content and tf.io.gfile.GFile
    csv_data = "index,mid,display_name\n0,/m/09x0r,Speech\n1,/m/04zqr,Music"
    with patch('tensorflow.io.gfile.GFile', mock_open(read_data=csv_data)) as mock_file, \
         patch('src.audio_analysis.analysis.librosa.load') as mock_librosa_load:

        # Instantiate the pipeline inside the patch context
        audio_pipeline = AudioAnalysisPipeline()

        # Assert that model and class names were loaded
        assert audio_pipeline.model is not None
        assert audio_pipeline.class_names == ['Speech', 'Music']

        # Mock what's needed for the analyze method
        mock_librosa_load.return_value = (np.zeros(16000), 16000)

        # Mock the model inference call
        scores = np.array([[0.9, 0.2]]) # High score for Speech
        # The model returns a tuple, with scores at the first position
        audio_pipeline.model.return_value = (tf.constant(scores), None, None)

        # Run analysis
        results = audio_pipeline.analyze('test.wav')

        # Assertions
        assert 'Speech' in results
        assert results['Speech'] > 0.8

@pytest.mark.skipif(tf is None, reason="TensorFlow is not installed")
@patch('src.audio_analysis.analysis.hub.load', side_effect=Exception("Model load error"))
def test_audio_pipeline_model_loading_failure(mock_hub_load, mock_config):
    """Test that analysis is skipped if the model fails to load."""
    audio_pipeline = AudioAnalysisPipeline()
    assert audio_pipeline.model is None
    results = audio_pipeline.analyze('dummy_audio.wav')
    assert results == {}