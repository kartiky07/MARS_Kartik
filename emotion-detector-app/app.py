import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import pickle
from audio_recorder_streamlit import audio_recorder
import wave

# 🎯 Load pre-trained model and label encoder
emotion_model = tf.keras.models.load_model("emotion_model.h5")
with open("label_encoder.pkl", "rb") as le_file:
    emotion_labels = pickle.load(le_file)

# 🎼 Extract MFCC features from a WAV file
def get_audio_features(path, sample_rate=22050, n_mfcc=40):
    signal, _ = librosa.load(path, sr=sample_rate)
    mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=n_mfcc)
    return mfccs.T

# 🔍 Predict emotion from audio file
def classify_emotion(path):
    features = get_audio_features(path)
    padded_input = tf.keras.preprocessing.sequence.pad_sequences(
        [features], padding='post', dtype='float32'
    )
    prediction = emotion_model.predict(padded_input)
    predicted_class = np.argmax(prediction)
    return emotion_labels.inverse_transform([predicted_class])[0]

# 🎤 Streamlit UI
st.set_page_config(page_title="Voice Emotion Detector", page_icon="🎧")
st.title("🎧 Real-Time Voice Emotion Detection")

input_mode = st.radio("Select input method:", ["📁 Upload Audio File", "🎙️ Record via Microphone"])

# 📁 Option 1: Upload WAV file
if input_mode == "📁 Upload Audio File":
    audio_file = st.file_uploader("Upload a WAV file below:", type="wav")
    if audio_file is not None:
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_file.read())
        st.audio("temp_audio.wav", format="audio/wav")
        result = classify_emotion("temp_audio.wav")
        st.success(f"🧠 Detected Emotion: *{result}*")

# 🎙️ Option 2: Record Audio Live
elif input_mode == "🎙️ Record via Microphone":
    st.info("Click the mic button to begin recording. It will stop after 3 seconds of silence.")
    
    recorded_audio = audio_recorder(
        energy_threshold=(-1.0, 1.0),
        pause_threshold=3.0,
        sample_rate=22050,
        text="",
        recording_color="#e0702d",
        neutral_color="#3b9c68"
    )
    
    if recorded_audio:
        with open("live_recording.wav", "wb") as out:
            out.write(recorded_audio)
        st.audio(recorded_audio, format="audio/wav")
        result = classify_emotion("live_recording.wav")
        st.success(f"🧠 Detected Emotion: *{result}*")