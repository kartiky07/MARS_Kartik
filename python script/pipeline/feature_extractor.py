import librosa
import numpy as np

# Extracts MFCC features from an audio file for LSTM input
def extract_features(file_path, sr=16000, n_mfcc=40):
    # Load audio
    audio, _ = librosa.load(file_path, sr=sr)

    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    # Transpose to shape (time_steps, n_mfcc)
    return mfcc.T
