import csv
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import os
import logging
from typing import Dict, Any, List
from tqdm import tqdm
from src.config import config

class AudioAnalysisPipeline:
    """
    A pipeline for analyzing audio files to classify sounds using YAMNet.
    The model is loaded once during initialization for efficiency.
    """
    def __init__(self):
        """
        Initializes the audio analysis pipeline by loading the model, class names, and config.
        """
        audio_config = config.get('audio_analysis', {})
        yamnet_url = audio_config.get('yamnet_url', 'https://tfhub.dev/google/yamnet/1')
        self.audio_score_threshold = audio_config.get('audio_score_threshold', 0.3)

        self.model = None
        self.class_names = []

        try:
            # Load the YAMNet model from TensorFlow Hub
            self.model = hub.load(yamnet_url)
            logging.info(f"Loaded YAMNet model from {yamnet_url}")

            # Load the class names
            class_map_path = self.model.class_map_path().numpy()
            self.class_names = self._class_names_from_csv(class_map_path)
            logging.info("Loaded YAMNet class names.")

        except Exception as e:
            logging.error(f"Failed to load YAMNet model or class names: {e}")

    @staticmethod
    def _class_names_from_csv(class_map_csv_text: str) -> List[str]:
        """Returns list of class names corresponding to score vector."""
        class_names = []
        with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                class_names.append(row[2])
        return class_names

    def analyze(self, wav_file_path: str) -> Dict[str, float]:
        """
        Analyzes an audio file to classify sounds using the pre-loaded YAMNet model.

        Args:
            wav_file_path (str): The path to the WAV audio file.

        Returns:
            dict: A dictionary of the top detected sounds and their scores.
        """
        if not self.model or not self.class_names:
            logging.error("YAMNet model or class names not loaded. Cannot analyze audio.")
            return {}

        try:
            waveform, _ = librosa.load(wav_file_path, sr=16000, mono=True)
        except Exception as e:
            logging.error(f"Failed to load audio file {wav_file_path}: {e}")
            return {}

        try:
            scores, _, _ = self.model(waveform)
            if hasattr(scores, 'numpy'):
                scores = scores.numpy()
        except Exception as e:
            logging.error(f"Failed during model inference: {e}")
            return {}

        mean_scores = np.mean(scores, axis=0)
        top_indices = np.argsort(mean_scores)[-5:]

        results = {}
        for i in reversed(top_indices):
            score = mean_scores[i]
            if score >= self.audio_score_threshold:
                results[self.class_names[i]] = score
                
        logging.info(f"Audio analysis complete. Detected {len(results)} sound types above threshold.")
        return results

if __name__ == '__main__':
    # This part is for direct testing of the script
    
    audio_pipeline = AudioAnalysisPipeline()
    
    if audio_pipeline.model:
        print("--- Testing with sine wave ---")
        sine_results = audio_pipeline.analyze('data/audio/sine_wave_1000hz.wav')
        if sine_results:
            for sound, score in sine_results.items():
                print(f"- {sound}: {score:.3f}")

        print("\n--- Testing with white noise ---")
        noise_results = audio_pipeline.analyze('data/audio/white_noise.wav')
        if noise_results:
            for sound, score in noise_results.items():
                print(f"- {sound}: {score:.3f}")
    else:
        print("Could not run audio analysis due to model loading failure.")