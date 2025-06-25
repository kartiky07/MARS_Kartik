
# mars_project
# **🎧 Emotion Detection from Audio using CNN-LSTM with Attention**

## **📌 Project Overview**
This project detects human emotions from speech audio using a deep learning model that combines **CNN**, **LSTM**, and an **Attention Mechanism**. It features a simple and interactive **web interface built with Streamlit**.

---

## **🎯 Objective**
To classify audio samples into emotional categories (e.g., Angry, Calm, Sad, etc.) using deep learning on MFCC-based features extracted from speech signals.

---

## **📁 Dataset: RAVDESS**
We use the **Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)** for training and testing.

- **Source:** Zenodo.org  
- **Link:** [https://zenodo.org/record/1188976](https://zenodo.org/record/1188976)
- Contains 24 professional actors (12 male, 12 female) vocalizing 8 emotions.
- Only **speech audio files** were used (no song files).
- The `surprised` class was **removed** to improve performance and address class imbalance.

---

---

## **🔍 Features Extracted**
We extract and use the following features from each audio clip using Librosa:
- **MFCCs (Mel-Frequency Cepstral Coefficients)**
- **Delta MFCCs (1st-order derivative)**
- **Delta-Delta MFCCs (2nd-order derivative)**
- All features are concatenated to form a `(time_steps, features)` matrix.

---

## **🧠 Model Architecture**
- **CNN Layer(s)**: For spatial feature extraction
- **LSTM Layer(s)**: To capture temporal dependencies
- **Attention Layer**: To focus on important time steps in audio
- **Dense Layer**: For final classification

---

## *🌐 Streamlit Web App*
💻 How to Run
bash
Copy
Edit
cd emotion-detector-app
##*streamlit run app.py *
Once launched, open the displayed URL (usually http://localhost:8501) in your browser.
# mars-project-2
# Kartik_mars
# Kartik_mars
# MARS_Kartik
