import csv
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import os
import logging
from typing import Dict, Any, Optional
from tqdm import tqdm

# Function to load class names from the CSV file provided by YAMNet
def class_names_from_csv(class_map_csv_text):
    """Returns list of class names corresponding to score vector."""
    class_names = []
    with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            class_names.append(row[2])
    return class_names

def analyze_audio(wav_file_path: str, config: Optional[Dict[Any, Any]] = None) -> Dict[str, float]:
    """
    Analyzes an audio file to classify sounds using the YAMNet model.

    Args:
        wav_file_path (str): The path to the WAV audio file.
        config (dict, optional): Configuration dictionary.

    Returns:
        dict: A dictionary of the top 5 detected sounds and their scores.
    """
    if config is None:
        config = {}
    
    # Get configuration values with defaults
    yamnet_url = config.get('models', {}).get('yamnet_url', 'https://tfhub.dev/google/yamnet/1')
    audio_score_threshold = config.get('thresholds', {}).get('audio_score_threshold', 0.3)
    
    try:
        # Load the YAMNet model from TensorFlow Hub
        model = hub.load(yamnet_url)
        logging.info(f"Loaded YAMNet model from {yamnet_url}")
    except Exception as e:
        logging.error(f"Failed to load YAMNet model: {e}")
        return {}

    try:
        # Load the class names
        class_map_path = model.class_map_path().numpy()
        class_names = class_names_from_csv(class_map_path)
    except Exception as e:
        logging.error(f"Failed to load class names: {e}")
        return {}

    try:
        # Load the audio file and resample it to 16kHz
        logging.info(f"Loading audio file: {wav_file_path}")
        waveform, sample_rate = librosa.load(wav_file_path, sr=16000, mono=True)
        logging.info(f"Audio loaded. Duration: {len(waveform)/sample_rate:.2f} seconds")
    except Exception as e:
        logging.error(f"Failed to load audio file {wav_file_path}: {e}")
        return {}

    try:
        # Run the model with progress bar
        logging.info("Running YAMNet inference...")
        pbar = tqdm(total=100, desc="Processing audio", unit="%")
        
        scores, embeddings, spectrogram = model(waveform)
        if hasattr(scores, 'numpy'):
            scores = scores.numpy()
        
        pbar.update(100)
        pbar.close()
    except Exception as e:
        logging.error(f"Failed during model inference: {e}")
        return {}

    try:
        # Get the top 5 predictions
        mean_scores = np.mean(scores, axis=0)
        top5_indices = np.argsort(mean_scores)[-5:]

        # Create a dictionary of the top 5 results above threshold
        results = {}
        for i in reversed(top5_indices):
            score = mean_scores[i]
            if score >= audio_score_threshold:
                results[class_names[i]] = score
                
        logging.info(f"Audio analysis complete. Detected {len(results)} sound types above threshold.")
    except Exception as e:
        logging.error(f"Failed during result processing: {e}")
        return {}

    return results


if __name__ == '__main__':
    # This part is for direct testing of the script
    import yaml
    
    # Load config for testing
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except:
        config = {}
    
    print("--- Testing with sine wave ---")
    sine_results = analyze_audio('data/audio/sine_wave_1000hz.wav', config)
    if sine_results:
        for sound, score in sine_results.items():
            print(f"- {sound}: {score:.3f}")

    print("\n--- Testing with white noise ---")
    noise_results = analyze_audio('data/audio/white_noise.wav', config)
    if noise_results:
        for sound, score in noise_results.items():
            print(f"- {sound}: {score:.3f}")
