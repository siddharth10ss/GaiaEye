import csv
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import os

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

def analyze_audio(wav_file_path):
    """
    Analyzes an audio file to classify sounds using the YAMNet model.

    Args:
        wav_file_path (str): The path to the WAV audio file.

    Returns:
        dict: A dictionary of the top 5 detected sounds and their scores.
    """
    # Load the YAMNet model from TensorFlow Hub
    model = hub.load('https://tfhub.dev/google/yamnet/1')

    # Load the class names
    class_map_path = model.class_map_path().numpy()
    class_names = class_names_from_csv(class_map_path)

    # Load the audio file and resample it to 16kHz
    waveform, sample_rate = librosa.load(wav_file_path, sr=16000, mono=True)

    # Run the model
    scores, embeddings, spectrogram = model(waveform)
    scores = scores.numpy()

    # Get the top 5 predictions
    top5_indices = np.argsort(np.mean(scores, axis=0))[-5:]

    # Create a dictionary of the top 5 results
    results = {}
    for i in reversed(top5_indices):
        results[class_names[i]] = np.mean(scores, axis=0)[i]

    return results


if __name__ == '__main__':
    # This part is for direct testing of the script
    print("--- Testing with sine wave ---")
    sine_results = analyze_audio('data/audio/sine_wave_1000hz.wav')
    if sine_results:
        for sound, score in sine_results.items():
            print(f"- {sound}: {score:.3f}")

    print("\n--- Testing with white noise ---")
    noise_results = analyze_audio('data/audio/white_noise.wav')
    if noise_results:
        for sound, score in noise_results.items():
            print(f"- {sound}: {score:.3f}")