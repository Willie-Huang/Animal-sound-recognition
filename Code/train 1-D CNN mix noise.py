import os
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
import json
import numpy as np
import librosa
from joblib import dump
from sklearn.preprocessing import LabelEncoder
from keras import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from keras.utils import pad_sequences
from keras import Input
from keras.layers import BatchNormalization, Dropout
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
import csv, random

# ---------- Path and hyperparameters ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = '/Users/pamachi/Desktop/Semester2/ELEC5305 Acoustic/Final assignment/Sound-Animal-Recognition-main/Animals_Sounds'
TRAIN_DIR = os.path.join(DATA_DIR, 'Training')
TEST_DIR  = os.path.join(DATA_DIR, 'Test')
MODEL_DIR = os.path.join(BASE_DIR, 'model')           # Model and configuration output directory
os.makedirs(MODEL_DIR, exist_ok=True)

N_MELS = 128
MAX_LENGTH = 500      # Time step length (number of frames)
EPOCHS = 20
BATCH_SIZE = 32
SR = 16000
N_FFT = 1024
HOP = 512

# ---------- wind-noise augmentation settings ----------
WIND_NOISE_DIR = '/Users/pamachi/Desktop/Semester2/ELEC5305 Acoustic/Final assignment/Sound-Animal-Recognition-main/Animals_Sounds/Noise/wind'
AUGMENT_WIND = True         # Only valid for TRAIN; no augmentation for TEST
AUG_PER_CLEAN = 4           # Generate 4 unmber of noisy versions for each clean sample
SNR_CANDIDATES = [15, 10, 5, 20, 0]
SNR_WEIGHTS    = [3,  4,  3,  1,  0.3]   # Make 10-15 dB more common and 0 dB rare
PREEMPH_ALPHA  = 0.95       # Mild pre-emphasis suppresses low-frequency wind noise
SEED = 42
random.seed(SEED); np.random.seed(SEED)

# ---------- date loading ----------
def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2) + 1e-12))

def _preemphasis(x: np.ndarray, alpha: float = PREEMPH_ALPHA) -> np.ndarray:
    if alpha <= 0.0:
        return x
    y = np.copy(x)
    y[1:] = y[1:] - alpha * y[:-1]
    return y

def _choose_chunk(noise: np.ndarray, out_len: int) -> np.ndarray:
    if len(noise) >= out_len:
        start = random.randint(0, len(noise) - out_len)
        return noise[start:start+out_len]
    rep = int(np.ceil(out_len / max(1, len(noise))))
    return np.tile(noise, rep)[:out_len]

def _mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    n = _choose_chunk(noise, len(clean))
    n = _preemphasis(n, PREEMPH_ALPHA)
    Ps = _rms(clean)**2
    Pn = _rms(n)**2
    k  = np.sqrt(Ps / (Pn * (10 ** (snr_db / 10.0))))
    mix = clean + k * n
    peak = np.max(np.abs(mix)) + 1e-9
    if peak > 1.0:
        mix = mix / peak * 0.98
    return mix

def _to_logmel(y: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP, center=True)
    log_mel = librosa.power_to_db(mel, ref=np.max).T
    mu = log_mel.mean(axis=0, keepdims=True)
    sigma = log_mel.std(axis=0, keepdims=True) + 1e-6
    return (log_mel - mu) / sigma

def _list_audio_files(root: str):
    for animal_folder in sorted(os.listdir(root)):
        animal_path = os.path.join(root, animal_folder)
        if not os.path.isdir(animal_path):
            continue
        for filename in sorted(os.listdir(animal_path)):
            if filename.lower().endswith(('.wav', '.ogg')):
                yield animal_folder, os.path.join(animal_path, filename)

def _load_noise_bank(noise_dir: str):
    noise_files = []
    if not os.path.isdir(noise_dir):
        print(f"[WARN] Noise dir not found: {noise_dir}")
        return noise_files
    for fn in sorted(os.listdir(noise_dir)):
        if fn.lower().endswith(('.wav', '.ogg')):
            fp = os.path.join(noise_dir, fn)
            try:
                y, _ = librosa.load(fp, sr=SR, mono=True)
                if len(y) > 0:
                    noise_files.append((fp, y))
            except Exception as e:
                print(f"[WARN] Failed to load noise {fp}: {e}")
    return noise_files

def load_audio_files_clean(data_dir: str, max_length: int = 500):
    audio_data, labels, file_paths = [], [], []
    unsupported = 0
    for animal_folder, file_path in _list_audio_files(data_dir):
        try:
            y, _ = librosa.load(file_path, sr=SR)
            log_mel = _to_logmel(y)
            audio_data.append(log_mel)
            labels.append(animal_folder)
            file_paths.append(file_path)
        except Exception as e:
            print(f"[WARN] Failed to load {file_path}: {e}")
    if not audio_data:
        raise ValueError(f"No audio files were loaded from {data_dir}.")
    X = pad_sequences(audio_data, maxlen=max_length, padding='post', truncating='post', dtype='float32')
    return np.asarray(X), labels, file_paths

def load_audio_files_with_wind(data_dir: str, noise_dir: str, max_length: int = 500):
    audio_data, labels, file_paths = [], [], []
    noise_bank = _load_noise_bank(noise_dir) if AUGMENT_WIND else []
    if AUGMENT_WIND:
        print(f"[INFO] Wind noise files: {len(noise_bank)} from {noise_dir}")
        if len(noise_bank) == 0:
            print("[WARN] No noise found, falling back to clean-only loading.")

    added_noisy = 0
    for animal_folder, file_path in _list_audio_files(data_dir):
        try:
            clean, _ = librosa.load(file_path, sr=SR)
            # 1) clean sample
            log_mel_clean = _to_logmel(clean)
            audio_data.append(log_mel_clean)
            labels.append(animal_folder)
            file_paths.append(file_path)

            # 2) add noise
            if AUGMENT_WIND and len(noise_bank) > 0:
                for k in range(AUG_PER_CLEAN):
                    nf, noise = random.choice(noise_bank)
                    snr = random.choices(SNR_CANDIDATES, weights=SNR_WEIGHTS, k=1)[0]
                    mixed = _mix_at_snr(clean, noise, snr)
                    log_mel_noisy = _to_logmel(mixed)
                    audio_data.append(log_mel_noisy)
                    labels.append(animal_folder)
                    file_paths.append(f"{file_path}#wind_snr{snr}_{k}")
                    added_noisy += 1
        except Exception as e:
            print(f"[WARN] Failed to load {file_path}: {e}")

    if not audio_data:
        raise ValueError(f"No audio files were loaded from {data_dir}.")
    X = pad_sequences(audio_data, maxlen=max_length, padding='post', truncating='post', dtype='float32')
    print(f"[INFO] Wind-aug: added noisy samples = {added_noisy}, clean samples = {len(labels) - added_noisy}")
    return np.asarray(X), labels, file_paths

print("[INFO] Loading TRAIN from:", TRAIN_DIR)
X_train, y_train_names, train_files = load_audio_files_with_wind(TRAIN_DIR, WIND_NOISE_DIR, max_length=MAX_LENGTH)

print("[INFO] Loading TEST  from:", TEST_DIR)
X_test,  y_test_names,  test_files  = load_audio_files_clean(TEST_DIR,  max_length=MAX_LENGTH)

# Label Encoding
le = LabelEncoder()
le.fit(sorted(set(y_train_names)))
num_classes = len(le.classes_)
print("[INFO] Classes (from TRAIN):", list(le.classes_))

y_train = le.transform(y_train_names)

# Ensure that the test set categories are all in the training categories
unknown = [c for c in set(y_test_names) if c not in le.classes_]
if unknown:
    raise ValueError(f"Classes in TEST not seen in TRAIN: {unknown}")
y_test = le.transform(y_test_names)

# Class distribution and class weights
counts = Counter(y_train_names)
print("[INFO] Train class counts:", dict(counts))
class_weight = {}
total = len(y_train_names)
k = len(counts)
for cls_name, cnt in counts.items():
    cls_idx = int(le.transform([cls_name])[0])
    class_weight[cls_idx] = total / (k * cnt)
print("[INFO] Class weights:", class_weight)

# ---------- model ----------
model = Sequential([
    Input(shape=(MAX_LENGTH, N_MELS)),
    Conv1D(32, 3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.3),

    Conv1D(64, 3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(num_classes, activation='softmax'),
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("[INFO] Training (no validation; test will be used once after training)...")
model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight,
    shuffle=True,
    verbose=2
)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"[INFO] Test Accuracy: {acc:.4f}")

# ---------- diagnostics ----------
# 1) Console text reports
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
report_txt = classification_report(y_test, y_pred, target_names=list(le.classes_))
print("[INFO] Classification Report:\n", report_txt)

# 2) Save classification report to CSV
report_csv = os.path.join(MODEL_DIR, "classification_report.csv")
with open(report_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["label", "precision", "recall", "f1-score", "support"])
    rpt = classification_report(y_test, y_pred, target_names=list(le.classes_), output_dict=True)
    for cls in le.classes_:
        row = rpt[cls]
        writer.writerow([cls, row["precision"], row["recall"], row["f1-score"], row["support"]])
    writer.writerow(["accuracy", "", "", rpt["accuracy"], ""])
print("[INFO] Saved:", report_csv)

# 3) Save confusion matrix to CSV
cm = confusion_matrix(y_test, y_pred)
cm_csv = os.path.join(MODEL_DIR, "confusion_matrix.csv")
np.savetxt(cm_csv, cm, delimiter=",", fmt="%d")
print("[INFO] Saved:", cm_csv)

# 4) Sample-by-sample prediction details (file path/true value/prediction/confidence)
probs = model.predict(X_test, verbose=0)
pred_labels = le.inverse_transform(y_pred)
pred_csv = os.path.join(MODEL_DIR, "predictions_detail.csv")
with open(pred_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["file", "true_label", "pred_label", "pred_confidence"])
    for i, fp in enumerate(test_files):
        writer.writerow([fp, y_test_names[i], pred_labels[i], float(np.max(probs[i]))])
print("[INFO] Saved:", pred_csv)

# ---------- save ----------
model_path = os.path.join(MODEL_DIR, "animal_sound_model.keras")  # Keras3 Native format
label_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
config_path = os.path.join(MODEL_DIR, "config.json")

model.save(model_path)
dump(le, label_path)
with open(config_path, "w") as f:
    json.dump({
        "MAX_LENGTH": MAX_LENGTH,
        "N_MELS": N_MELS,
        "classes": list(le.classes_),
        "SR": SR, "N_FFT": N_FFT, "HOP": HOP,
        "DATA_DIR": DATA_DIR, "TRAIN_DIR": TRAIN_DIR, "TEST_DIR": TEST_DIR
    }, f, ensure_ascii=False, indent=2)

print("[INFO] Saved:")
print("  -", model_path)
print("  -", label_path)
print("  -", config_path)
