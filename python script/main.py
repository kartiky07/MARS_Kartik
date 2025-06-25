from pipeline.data_loader import get_ravdess_files
from pipeline.feature_extractor import extract_features
from pipeline.model import build_lstm_model
from pipeline.trainer import train_model
from pipeline.evaluator import evaluate_model, plot_history

from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import tensorflow as tf
import pandas as pd
import numpy as np

# Configuration
SPEECH_DIR = "data/Audio_Speech_Actors_01-24"
SONG_DIR = "data/Audio_Song_Actors_01-24"
SELECTED_EMOTIONS = ["happy", "sad", "angry", "calm", "fearful"]

# 🔹 1. Load files
file_paths_speech, labels_speech = get_ravdess_files(SPEECH_DIR, SELECTED_EMOTIONS)
file_paths_song, labels_song = get_ravdess_files(SONG_DIR, SELECTED_EMOTIONS)

# Combine both speech and song
all_file_paths = file_paths_speech + file_paths_song
all_labels = labels_speech + labels_song

# 🔹 2. Balance the dataset
df = pd.DataFrame({'file': all_file_paths, 'label': all_labels})
print("Before balancing:\n", df['label'].value_counts())

# Oversample to balance classes
max_count = df['label'].value_counts().max()
df_balanced = df.groupby('label').apply(lambda x: resample(x, replace=True, n_samples=max_count, random_state=42)).reset_index(drop=True)

# Extract balanced data
file_paths = df_balanced['file'].tolist()
labels = df_balanced['label'].tolist()
print("After balancing:\n", pd.Series(labels).value_counts())

# 🔹 3. Extract features
print("🔄 Extracting features...")
X = [extract_features(f) for f in file_paths]

# 🔹 4. Pad sequences
X = pad_sequences(X, padding='post', dtype='float32')  # shape: (samples, time_steps, 40)

# 🔹 5. Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)
y_onehot = tf.keras.utils.to_categorical(y_encoded)

# 🔹 6. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# 🔹 7. Build and train model
input_shape = X_train.shape[1:]  # (time_steps, 40)
model = build_lstm_model(input_shape, len(SELECTED_EMOTIONS))
history = train_model(model, X_train, y_train)

# 🔹 8. Evaluate
evaluate_model(model, X_test, y_test, label_encoder.classes_)
plot_history(history)
