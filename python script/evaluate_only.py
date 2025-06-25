# evaluate_only.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from pipeline.data_loader import get_ravdess_files
from pipeline.feature_extractor import extract_features
from pipeline.evaluator import evaluate_model

# Configuration
SPEECH_DIR = "data/Audio_Speech_Actors_01-24"
SONG_DIR = "data/Audio_Song_Actors_01-24"
SELECTED_EMOTIONS = ["happy", "sad", "angry", "calm", "fearful"]
MODEL_PATH = "saved_models/emotion_model.h5"

# Load file paths and labels
file_paths_speech, labels_speech = get_ravdess_files(SPEECH_DIR, SELECTED_EMOTIONS)
file_paths_song, labels_song = get_ravdess_files(SONG_DIR, SELECTED_EMOTIONS)

file_paths = file_paths_speech + file_paths_song
labels = labels_speech + labels_song

# Feature extraction
print("📦 Extracting features...")
X_seq = [extract_features(f, mode='lstm') for f in file_paths]
X = pad_sequences(X_seq, padding='post', dtype='float32')  # Shape: (samples, timesteps, features)

# Encode emotion labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)
y_onehot = tf.keras.utils.to_categorical(y_encoded)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# Load trained model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
model = load_model(MODEL_PATH)

# Evaluate the model
evaluate_model(model, X_test, y_test, label_encoder.classes_)
