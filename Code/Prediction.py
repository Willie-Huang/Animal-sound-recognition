import os
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Mute most TF logs
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass

import argparse
import json
import numpy as np
import librosa
from joblib import load
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import pad_sequences

# ---------- Path ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
IMAGE_DIR = os.path.join(BASE_DIR, 'animal_pictures')

DATA_DIR  = '/Users/pamachi/Desktop/Semester2/ELEC5305 Acoustic/Final assignment/Sound-Animal-Recognition-main/Animals_Sounds'

MODEL_PATH = os.path.join(MODEL_DIR, 'animal_sound_model.keras')
LABEL_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')
CONFIG_PATH = os.path.join(MODEL_DIR, 'config.json')

# ---------- Loading models and configurations ----------
if not (os.path.exists(MODEL_PATH) and os.path.exists(LABEL_PATH) and os.path.exists(CONFIG_PATH)):
    raise FileNotFoundError("Model files not found. Please run train.py first.")

model = load_model(MODEL_PATH)
le = load(LABEL_PATH)
with open(CONFIG_PATH) as f:
    cfg = json.load(f)
MAX_LENGTH = int(cfg["MAX_LENGTH"])
N_MELS = int(cfg["N_MELS"])
SR = int(cfg.get("SR", 16000))
N_FFT = int(cfg.get("N_FFT", 1024))
HOP = int(cfg.get("HOP", 512))

# ---------- Prediction ----------
def predict_animal_sound(audio_file_path: str) -> str:
    y, _ = librosa.load(audio_file_path, sr=SR)  # Unified to the training sampling rate
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP, center=True
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).T
    # Normalize per sample
    mu = log_mel.mean(axis=0, keepdims=True)
    sigma = log_mel.std(axis=0, keepdims=True) + 1e-6
    log_mel = (log_mel - mu) / sigma
    X = pad_sequences([log_mel], maxlen=MAX_LENGTH, padding='post',
                      truncating='post', dtype='float32')
    probs = model.predict(X, verbose=0)[0]
    idx = int(np.argmax(probs))
    return le.inverse_transform([idx])[0]

def show_animal_image(label: str):
    img_path = os.path.join(IMAGE_DIR, f"{label}.jpeg")
    if os.path.exists(img_path):
        img = Image.open(img_path)
        plt.imshow(img); plt.axis('off'); plt.title(f"Predicted: {label}")
        plt.show()
    else:
        print(f"[INFO] No image found for label: {label}")

def find_first_audio_file(root: str):
    # Recursively search for the first .wav/.ogg file in the dataset directory for automatic fallback prediction when PyAudio is not available.
    for dirpath, _, filenames in os.walk(root):
        for fn in sorted(filenames):
            if fn.lower().endswith(('.wav', '.ogg')):
                return os.path.join(dirpath, fn)
    return None

# ---------- Microphone recording ----------
def record_or_fallback_predict():
    """
    Behavior Description:
    1) If PyAudio is installed and microphone permission is granted, record 5 seconds and save as temp.wav. Then perform the recognition.
    2) If PyAudio is not installed or unavailable, automatically select an audio from the Animals_Sounds dataset for prediction (without interruption or error).
    """
    # Try Microphone Path
    try:
        import speech_recognition as sr
        try:
            import pyaudio  # detect PyAudio is useful
            have_pyaudio = True
        except Exception as e:
            have_pyaudio = False
            reason = e

        if have_pyaudio:
            rec = sr.Recognizer()
            with sr.Microphone() as source:
                print("[INFO] Detected PyAudio. Listening 5s via microphone...")
                audio = rec.listen(source, timeout=5)
            tmp = os.path.join(BASE_DIR, "temp.wav")
            with open(tmp, "wb") as f:
                f.write(audio.get_wav_data())

            label = predict_animal_sound(tmp)
            print("Predicted:", label)
            show_animal_image(label)
            return
        else:
            print("[INFO] PyAudio not available. Fallback to dataset. Reason:", reason)

    except Exception as mic_err:
        print("[INFO] Microphone path failed. Fallback to dataset. Reason:", mic_err)

    # Back-up: Pick an audio from the dataset
    sample = find_first_audio_file(DATA_DIR)
    if sample:
        print(f"[INFO] Using dataset sample: {sample}")
        label = predict_animal_sound(sample)
        print("Predicted:", label)
        show_animal_image(label)
    else:
        print("[WARN] No audio file found in dataset. Please pass --file /path/to/audio.wav")

# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=str, help="path to a wav/ogg file")
    args = ap.parse_args()

    if args.file:
        pred = predict_animal_sound(args.file)
        print("Predicted:", pred)
        show_animal_image(pred)
    else:
        test_dir = os.path.join(DATA_DIR, "Test")
        sample = find_first_audio_file(test_dir) or find_first_audio_file(DATA_DIR)
        if sample:
            pred = predict_animal_sound(sample)
            print("Predicted:", pred)
            show_animal_image(pred)
        else:
            print("[WARN] No audio file found in dataset. Please pass --file /path/to/audio.wav")
