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
import csv

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

# ---------- date loading ----------
def load_audio_files(data_dir: str, max_length: int = 500):
    audio_data = []
    labels = []
    file_paths = []
    unsupported = 0

    for animal_folder in sorted(os.listdir(data_dir)):
        animal_path = os.path.join(data_dir, animal_folder)
        if not os.path.isdir(animal_path):
            continue
        for filename in sorted(os.listdir(animal_path)):
            file_path = os.path.join(animal_path, filename)
            if not (filename.lower().endswith('.wav') or filename.lower().endswith('.ogg')):
                unsupported += 1
                continue
            try:
                y, sr = librosa.load(file_path, sr=SR)
                mel = librosa.feature.melspectrogram(
                    y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP, center=True
                )
                log_mel = librosa.power_to_db(mel, ref=np.max).T
                mu = log_mel.mean(axis=0, keepdims=True)
                sigma = log_mel.std(axis=0, keepdims=True) + 1e-6
                log_mel = (log_mel - mu) / sigma
                audio_data.append(log_mel)
                labels.append(animal_folder)
                file_paths.append(file_path)
            except Exception as e:
                print(f"[WARN] Failed to load {file_path}: {e}")

    print(f"Unsupported files skipped: {unsupported}")
    if not audio_data:
        raise ValueError(f"No audio files were loaded from {data_dir}.")

    X = pad_sequences(audio_data, maxlen=max_length, padding='post',
                      truncating='post', dtype='float32')
    return np.asarray(X), labels, file_paths

print("[INFO] Loading TRAIN from:", TRAIN_DIR)
X_train, y_train_names, train_files = load_audio_files(TRAIN_DIR, max_length=MAX_LENGTH)

print("[INFO] Loading TEST  from:", TEST_DIR)
X_test,  y_test_names,  test_files  = load_audio_files(TEST_DIR,  max_length=MAX_LENGTH)

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
