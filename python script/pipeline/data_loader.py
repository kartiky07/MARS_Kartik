import os

# Mapping of emotion codes to emotion labels
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# Reads files recursively from each actor's folder
def get_ravdess_files(base_dir, selected_emotions):
    file_paths = []
    labels = []

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".wav"):
                parts = file.split("-")
                emotion_code = parts[2]  # 3rd part is emotion
                emotion = EMOTION_MAP.get(emotion_code)
                if emotion in selected_emotions:
                    full_path = os.path.join(root, file)
                    file_paths.append(full_path)
                    labels.append(emotion)

    return file_paths, labels

