import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import os

from src.audio_analysis.analysis import analyze_audio, class_names_from_csv

@pytest.fixture
def mock_config():
    return {
        'models': {
            'yamnet_url': 'https://tfhub.dev/google/yamnet/1'
        },
        'thresholds': {
            'audio_score_threshold': 0.3
        }
    }

def test_analyze_audio_with_valid_input(mock_config):
    """Test audio analysis with valid input."""
    with patch('src.audio_analysis.analysis.hub.load') as mock_hub_load, \
         patch('src.audio_analysis.analysis.librosa.load') as mock_librosa_load, \
         patch('src.audio_analysis.analysis.class_names_from_csv') as mock_class_names_from_csv:

        # Mock the model and its components
        mock_model = MagicMock()
        mock_hub_load.return_value = mock_model
        mock_model.class_map_path.return_value.numpy.return_value = 'class_map.csv'
        mock_class_names_from_csv.return_value = ['Speech', 'Music', 'Dog', 'Cat', 'Bird']

        # Mock librosa.load
        mock_librosa_load.return_value = (np.zeros(16000), 16000)

        # Mock the model inference
        scores = np.array([[0.1, 0.2, 0.8, 0.5, 0.4]])
        mock_model.return_value = (scores, None, None)

        # Test the function
        results = analyze_audio('test.wav', mock_config)

        # Assertions
        assert isinstance(results, dict)
        assert 'Dog' in results
        assert results['Dog'] == 0.8

def test_analyze_audio_with_invalid_file(mock_config):
    """Test audio analysis with invalid file path."""
    with patch('src.audio_analysis.analysis.librosa.load') as mock_librosa_load:
        mock_librosa_load.side_effect = FileNotFoundError
        results = analyze_audio('nonexistent.wav', mock_config)
        assert isinstance(results, dict)
        assert len(results) == 0
