import os
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
import json
import numpy as np
import librosa
from joblib import dump
from sklearn.preprocessing import LabelEncoder
from keras import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense
from keras.utils import pad_sequences
from keras import Input
from keras.layers import BatchNormalization, Dropout
from collections import Counter
import csv, random

# ---------- Path and hyperparameters ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = '/Users/pamachi/Desktop/Semester2/ELEC5305 Acoustic/Final assignment/Sound-Animal-Recognition-main/Animals_Sounds'
TRAIN_DIR = os.path.join(DATA_DIR, 'Training')
TEST_DIR  = os.path.join(DATA_DIR, 'Test')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
os.makedirs(MODEL_DIR, exist_ok=True)

N_MELS = 128
MAX_LENGTH = 500
EPOCHS = 20
BATCH_SIZE = 32
SR = 16000
N_FFT = 1024
HOP = 512

# ---------- SSL & ProtoNet hyperparameters ----------
SSL_EPOCHS = 20
SSL_BATCH  = 64
SSL_TEMP   = 0.1
EMB_DIM    = 128
PROJ_DIM   = 64

PROTO_EPOCHS = 30
N_WAY   = 5
K_SHOT  = 5
Q_QUERY = 5
EPISODES_PER_EPOCH = 40

# Controlled augmentation & noise
SEED = 42
random.seed(SEED); np.random.seed(SEED)
AUG_SEED = 1234
SNR_LIST_DB = [5, 0, -5]
TIME_MASK_MAX = 30
FREQ_MASK_MAX = 16
TIME_SHIFT_FRAMES = 20

# ---------- data loading ----------
def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2) + 1e-12))

def _to_logmel(y: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP, center=True)
    log_mel = librosa.power_to_db(mel, ref=np.max).T
    mu = log_mel.mean(axis=0, keepdims=True)
    sigma = log_mel.std(axis=0, keepdims=True) + 1e-6
    return (log_mel - mu) / sigma

def _list_audio_files(root: str):
    for animal_folder in sorted(os.listdir(root)):
        animal_path = os.path.join(root, animal_folder)
        if not os.path.isdir(animal_path): continue
        for filename in sorted(os.listdir(animal_path)):
            if filename.lower().endswith(('.wav', '.ogg')):
                yield animal_folder, os.path.join(animal_path, filename)

def load_audio_files_clean(data_dir: str, max_length: int = 500):
    audio_data, labels, file_paths = [], [], []
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

print("[INFO] Loading TRAIN from:", TRAIN_DIR)
X_train, y_train_names, train_files = load_audio_files_clean(TRAIN_DIR, max_length=MAX_LENGTH)

print("[INFO] Loading TEST  from:", TEST_DIR)
X_test,  y_test_names,  test_files  = load_audio_files_clean(TEST_DIR,  max_length=MAX_LENGTH)

# Label Encoding
le = LabelEncoder()
le.fit(sorted(set(y_train_names)))
num_classes = len(le.classes_)
print("[INFO] Classes (from TRAIN):", list(le.classes_))

y_train = le.transform(y_train_names)

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

# ---------- Controlled augmentations ----------
rng = np.random.default_rng(AUG_SEED)

def time_shift(x: np.ndarray, max_frames: int = TIME_SHIFT_FRAMES):
    """
    Supports 2D (T,F) and 3D (B,T,F). Positive shift = shift right in time.
    """
    if max_frames <= 0:
        return x
    shift = int(rng.integers(-max_frames, max_frames + 1))
    if shift == 0:
        return x

    if x.ndim == 2:
        # (T,F)
        if shift > 0:
            return np.pad(x, ((shift, 0), (0, 0)), mode='constant')[:-shift]
        else:
            return np.pad(x, ((0, -shift), (0, 0)), mode='constant')[-shift:]
    elif x.ndim == 3:
        # (B,T,F)
        B, T, F = x.shape
        if shift > 0:
            y = np.pad(x, ((0,0), (shift,0), (0,0)), mode='constant')[:, :-shift, :]
        else:
            y = np.pad(x, ((0,0), (0,-shift), (0,0)), mode='constant')[:, -shift:, :]
        return y
    else:
        raise ValueError(f"time_shift expects 2D or 3D array, got shape {x.shape}")

def spec_mask(x: np.ndarray, time_max=TIME_MASK_MAX, freq_max=FREQ_MASK_MAX):
    """
    Time/freq masking. Supports 2D (T,F) and 3D (B,T,F). For 3D, mask per-sample.
    """
    m = x.copy()
    if x.ndim == 2:
        T, F = m.shape
        t = int(rng.integers(0, min(time_max, T))) if time_max > 0 else 0
        t0 = int(rng.integers(0, T - t + 1)) if t > 0 else 0
        if t > 0: m[t0:t0 + t, :] = 0.0

        f = int(rng.integers(0, min(freq_max, F))) if freq_max > 0 else 0
        f0 = int(rng.integers(0, F - f + 1)) if f > 0 else 0
        if f > 0: m[:, f0:f0 + f] = 0.0
        return m
    elif x.ndim == 3:
        B, T, F = m.shape
        for b in range(B):
            tb = int(rng.integers(0, min(time_max, T))) if time_max > 0 else 0
            t0b = int(rng.integers(0, T - tb + 1)) if tb > 0 else 0
            if tb > 0: m[b, t0b:t0b + tb, :] = 0.0

            fb = int(rng.integers(0, min(freq_max, F))) if freq_max > 0 else 0
            f0b = int(rng.integers(0, F - fb + 1)) if fb > 0 else 0
            if fb > 0: m[b, :, f0b:f0b + fb] = 0.0
        return m
    else:
        raise ValueError(f"spec_mask expects 2D or 3D array, got shape {x.shape}")

NOISE_DIR = os.path.join(DATA_DIR, 'Noise', 'wind')
def _list_noise_files(noise_dir: str):
    if not os.path.isdir(noi se_dir): return []
    files = [os.path.join(noise_dir, fn) for fn in sorted(os.listdir(noise_dir))
             if fn.lower().endswith(('.wav','.ogg'))]
    return files

NOISE_FILES = _list_noise_files(NOISE_DIR)
print(f"[INFO] Noise files available: {len(NOISE_FILES)} from {NOISE_DIR}")

def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    alpha = 10 ** (-snr_db/20.0)
    return (clean + alpha * noise) / (1.0 + alpha)

def augment_view(x: np.ndarray) -> np.ndarray:
    """
    Accepts 2D (T,F) or 3D (B,T,F). Applies time_shift then spec_mask.
    """
    v = time_shift(x)
    v = spec_mask(v)
    return v

# ---------- Encoder & Heads ----------
import tensorflow as tf
from tensorflow.keras import layers, Model

def build_encoder(input_shape=(MAX_LENGTH, N_MELS), emb_dim=EMB_DIM):
    inp = Input(shape=input_shape)
    x = Conv1D(64, 5, padding='same', activation='relu')(inp)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.2)(x)

    x = Conv1D(128, 5, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.2)(x)

    x = Conv1D(256, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.2)(x)

    x = layers.GlobalAveragePooling1D()(x)
    z = Dense(EMB_DIM)(x)
    z = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(z)
    return Model(inp, z, name="encoder")

def build_projection_head(emb_dim=EMB_DIM, proj_dim=PROJ_DIM):
    inp = Input(shape=(emb_dim,))
    x = Dense(emb_dim, activation='relu')(inp)
    out = Dense(proj_dim)(x)
    out = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(out)
    return Model(inp, out, name="proj_head")

encoder = build_encoder()
proj_head = build_projection_head()

# ---------- Stage A: Self-Supervised SimCLR pretraining ----------
def make_ssl_batch(X, batch_size):
    idx = rng.choice(len(X), size=batch_size, replace=len(X)<batch_size)
    x0 = X[idx]
    if len(NOISE_FILES) > 0:
        noise_idx = rng.choice(len(X), size=batch_size, replace=True)
        noise_samples = X[noise_idx]
        snr_db = rng.choice(SNR_LIST_DB)
        x0n = mix_at_snr(x0, noise_samples, snr_db)
    else:
        x0n = x0
    v1 = augment_view(x0n)
    v2 = augment_view(x0n)
    return v1.astype('float32'), v2.astype('float32')

@tf.function
def nt_xent_loss(z1, z2, temp=SSL_TEMP):
    z1 = tf.math.l2_normalize(z1, axis=-1)
    z2 = tf.math.l2_normalize(z2, axis=-1)
    z = tf.concat([z1, z2], axis=0)
    sim = tf.matmul(z, z, transpose_b=True)
    B = tf.shape(z1)[0]
    mask = tf.eye(2*B)
    sim = sim - 1e9*mask
    logits = sim / temp
    labels = tf.concat([tf.range(B, 2*B), tf.range(0, B)], axis=0)
    loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True))
    return loss

ssl_opt = tf.keras.optimizers.Adam(1e-3)
ssl_steps_per_epoch = max(1, len(X_train)//SSL_BATCH)

print(f"[INFO] SSL pretraining: epochs={SSL_EPOCHS}, steps/epoch={ssl_steps_per_epoch}")
for ep in range(SSL_EPOCHS):
    epoch_loss = []
    for _ in range(ssl_steps_per_epoch):
        v1, v2 = make_ssl_batch(X_train, SSL_BATCH)
        with tf.GradientTape() as tape:
            z1 = proj_head(encoder(v1, training=True), training=True)
            z2 = proj_head(encoder(v2, training=True), training=True)
            loss_ssl = nt_xent_loss(z1, z2)
        grads = tape.gradient(loss_ssl, encoder.trainable_variables + proj_head.trainable_variables)
        ssl_opt.apply_gradients(zip(grads, encoder.trainable_variables + proj_head.trainable_variables))
        epoch_loss.append(float(loss_ssl.numpy()))
    print(f"[SSL][{ep+1:02d}/{SSL_EPOCHS}] loss={np.mean(epoch_loss):.4f}")
print("[INFO] SSL pretraining done; freezing projection head.")
proj_head.trainable = False

# ---------- Stage B: Few-shot ProtoNet fine-tuning ----------
train_by_class = {}
for i, cname in enumerate(y_train_names):
    train_by_class.setdefault(cname, []).append(i)

rng = np.random.default_rng(AUG_SEED+1)

def sample_episode(n_way=N_WAY, k_shot=K_SHOT, q_query=Q_QUERY):
    classes = rng.choice(list(train_by_class.keys()), size=n_way, replace=False)
    Xs, Ys, Xq, Yq = [], [], [], []
    for ci, cname in enumerate(classes):
        idxs = train_by_class[cname]
        if len(idxs) < k_shot + q_query:
            choice = rng.choice(idxs, size=k_shot+q_query, replace=True)
        else:
            choice = rng.choice(idxs, size=k_shot+q_query, replace=False)
        sup = choice[:k_shot]; que = choice[k_shot:]
        Xs.append(X_train[sup]); Xq.append(X_train[que])
        Ys.append(np.full(k_shot, ci)); Yq.append(np.full(q_query, ci))
    Xs = np.concatenate(Xs, axis=0).astype('float32')
    Xq = np.concatenate(Xq, axis=0).astype('float32')
    Ys = np.concatenate(Ys, axis=0).astype('int32')
    Yq = np.concatenate(Yq, axis=0).astype('int32')
    return Xs, Ys, Xq, Yq

@tf.function
def prototypical_step(Xs, Ys, Xq, Yq, enc, opt):
    """
    Vectorized ProtoNet step without unsorted_segment_mean
    (uses one-hot aggregation -> matmul; MPS-friendly).
    """
    with tf.GradientTape() as tape:
        Es = enc(Xs, training=True)  # [Ns, d], Ns = n_way * k_shot
        Eq = enc(Xq, training=True)  # [Nq, d]

        # labels & n_way
        Ys = tf.cast(Ys, tf.int32)   # [Ns]
        Yq = tf.cast(Yq, tf.int32)   # [Nq]
        n_way = tf.cast(tf.reduce_max(Ys) + 1, tf.int32)

        # one-hot -> sum embeddings per class
        oh = tf.one_hot(Ys, depth=n_way, dtype=Es.dtype)      # [Ns, n_way]
        sum_emb = tf.matmul(oh, Es, transpose_a=True)         # [n_way, d]
        cnt = tf.reduce_sum(oh, axis=0)                       # [n_way]
        protos = sum_emb / tf.maximum(tf.expand_dims(cnt, -1), tf.constant(1.0, dtype=Es.dtype))  # [n_way, d]

        # squared Euclidean logits
        Eq2 = tf.reduce_sum(tf.square(Eq), axis=1, keepdims=True)                 # [Nq,1]
        P2  = tf.reduce_sum(tf.square(protos), axis=1, keepdims=True)             # [n_way,1]
        logits = - (Eq2 - 2.0 * tf.matmul(Eq, protos, transpose_b=True) + tf.transpose(P2))  # [Nq,n_way]

        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(Yq, logits, from_logits=True)
        )

    grads = tape.gradient(loss, enc.trainable_variables)
    opt.apply_gradients(zip(grads, enc.trainable_variables))

    preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
    acc = tf.reduce_mean(tf.cast(tf.equal(preds, Yq), tf.float32))
    return loss, acc

proto_opt = tf.keras.optimizers.Adam(1e-3)
print(f"[INFO] ProtoNet fine-tuning: epochs={PROTO_EPOCHS}, episodes/epoch={EPISODES_PER_EPOCH}")
for ep in range(PROTO_EPOCHS):
    losses, accs = [], []
    for _ in range(EPISODES_PER_EPOCH):
        Xs, Ys, Xq, Yq = sample_episode()
        l, a = prototypical_step(Xs, Ys, Xq, Yq, encoder, proto_opt)
        losses.append(float(l.numpy())); accs.append(float(a.numpy()))
    print(f"[PROTO][{ep+1:02d}/{PROTO_EPOCHS}] loss={np.mean(losses):.4f} acc={np.mean(accs):.3f}")
print("[INFO] ProtoNet fine-tuning done.")

# ---------- Stage C: Evaluation ----------
print("[INFO] Prototype evaluation on TEST using TRAIN as support")
emb_train = encoder.predict(X_train, verbose=0)
emb_test  = encoder.predict(X_test,  verbose=0)

protos_by_idx = {}
for cls_name in le.classes_:
    cls_idx = int(le.transform([cls_name])[0])
    mask = np.where(y_train == cls_idx)[0]
    protos_by_idx[cls_idx] = emb_train[mask].mean(axis=0, keepdims=True)

protos = np.concatenate([protos_by_idx[i] for i in range(len(le.classes_))], axis=0)

def proto_logits(Eq, protos):
    Eq2 = np.sum(Eq**2, axis=1, keepdims=True)
    P2  = np.sum(protos**2, axis=1, keepdims=True).T
    return - (Eq2 - 2*np.matmul(Eq, protos.T) + P2)

logits = proto_logits(emb_test, protos)
y_pred = np.argmax(logits, axis=1)
acc = (y_pred == y_test).mean()
print(f"[INFO] ProtoNet TEST Accuracy: {acc:.4f}")

# Linear probe sanity check
lin_clf = Sequential([Input(shape=(EMB_DIM,)), Dense(num_classes, activation='softmax')])
lin_clf.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
lin_clf.fit(emb_train, y_train, epochs=10, batch_size=64, verbose=0, class_weight=class_weight)
lp_loss, lp_acc = lin_clf.evaluate(emb_test, y_test, verbose=0)
print(f"[INFO] Linear Probe TEST Accuracy: {lp_acc:.4f}")

# ---------- diagnostics ----------
from sklearn.metrics import classification_report, confusion_matrix
report_txt = classification_report(y_test, y_pred, target_names=list(le.classes_))
print("[INFO] Classification Report (ProtoNet):\n", report_txt)

report_csv = os.path.join(MODEL_DIR, "classification_report_protonet.csv")
with open(report_csv, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["label","precision","recall","f1-score","support"])
    rpt = classification_report(y_test, y_pred, target_names=list(le.classes_), output_dict=True)
    for cls in le.classes_:
        row = rpt[cls]; w.writerow([cls, row["precision"], row["recall"], row["f1-score"], row["support"]])
    w.writerow(["accuracy","","",rpt["accuracy"],""])
print("[INFO] Saved:", report_csv)

cm = confusion_matrix(y_test, y_pred)
cm_csv = os.path.join(MODEL_DIR, "confusion_matrix_protonet.csv")
np.savetxt(cm_csv, cm, delimiter=",", fmt="%d")
print("[INFO] Saved:", cm_csv)

probs = tf.nn.softmax(logits, axis=-1).numpy()
pred_labels = le.inverse_transform(y_pred)
pred_csv = os.path.join(MODEL_DIR, "predictions_detail_protonet.csv")
with open(pred_csv, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["file","true_label","pred_label","pred_confidence"])
    for i, fp in enumerate(test_files):
        w.writerow([fp, y_test_names[i], pred_labels[i], float(np.max(probs[i]))])
print("[INFO] Saved:", pred_csv)

# ---------- save ----------
enc_path   = os.path.join(MODEL_DIR, "encoder_ssl_protonet.keras")
proj_path  = os.path.join(MODEL_DIR, "proj_head_ssl.keras")
lin_path   = os.path.join(MODEL_DIR, "linear_probe.keras")
label_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
config_path= os.path.join(MODEL_DIR, "config.json")

encoder.save(enc_path)
proj_head.save(proj_path)
lin_clf.save(lin_path)
dump(le, label_path)
with open(config_path, "w") as f:
    json.dump({
        "MAX_LENGTH": MAX_LENGTH, "N_MELS": N_MELS, "classes": list(le.classes_),
        "SR": SR, "N_FFT": N_FFT, "HOP": HOP,
        "SSL_EPOCHS": SSL_EPOCHS, "PROTO_EPOCHS": PROTO_EPOCHS,
        "N_WAY": N_WAY, "K_SHOT": K_SHOT, "Q_QUERY": Q_QUERY
    }, f, ensure_ascii=False, indent=2)

print("[INFO] Saved:")
print("  -", enc_path)
print("  -", proj_path)
print("  -", lin_path)
print("  -", label_path)
print("  -", config_path)
