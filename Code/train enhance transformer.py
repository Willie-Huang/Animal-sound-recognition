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
from keras import layers, Model
from collections import Counter
import csv, random
import tensorflow as tf


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
SSL_TEMP   = 0.3
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
TIME_MASK_MAX = 30
FREQ_MASK_MAX = 16
TIME_SHIFT_FRAMES = 20

# ---------- Transformer encoder utils (inserted) ----------
class PositionalEncoding(layers.Layer):
    def __init__(self, d_model: int, max_len: int = 4000, learnable: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.learnable = learnable
        if learnable:
            self.pe = self.add_weight(
                "pe", shape=(max_len, d_model), initializer="zeros", trainable=True
            )
        else:
            pos = np.arange(max_len)[:, None]
            i = np.arange(d_model)[None, :]
            angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
            angles = pos * angle_rates
            pe = np.zeros((max_len, d_model), dtype=np.float32)
            pe[:, 0::2] = np.sin(angles[:, 0::2])
            pe[:, 1::2] = np.cos(angles[:, 1::2])
            self.pe = tf.constant(pe, dtype=tf.float32)

    def call(self, x):
        # x: [B, T, D]
        T = tf.shape(x)[1]
        return x + self.pe[:T] if not self.learnable else x + self.pe[:T]

class TransformerBlock(layers.Layer):
    def __init__(self, d_model: int, num_heads: int, mlp_dim: int, drop: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn  = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model, dropout=drop)
        self.drop1 = layers.Dropout(drop)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        self.ffn   = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation="relu"),
            layers.Dropout(drop),
            layers.Dense(d_model),
        ])
        self.drop2 = layers.Dropout(drop)

    def call(self, x, training=False):
        # MHA
        h = self.norm1(x)
        h = self.attn(h, h, training=training)
        x = x + self.drop1(h, training=training)
        # FFN
        h = self.norm2(x)
        h = self.ffn(h, training=training)
        x = x + self.drop2(h, training=training)
        return x

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

print("[INFO] Loading TEST from:", TEST_DIR)
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

def _rand_gain(x: np.ndarray, low=0.8, high=1.25):
    g = rng.uniform(low, high)
    return x * g

def augment_view(x: np.ndarray) -> np.ndarray:
    """
    Accepts 2D (T,F) or 3D (B,T,F). Applies time_shift then spec_mask.
    """
    v = time_shift(x)
    v = spec_mask(v)
    return v

# ---------- Encoder & Heads ----------
def build_encoder(
    input_shape=(MAX_LENGTH, N_MELS),
    emb_dim: int = EMB_DIM,
    d_model: int = 128,
    num_heads: int = 4,
    mlp_dim: int = 256,
    num_layers: int = 4,
    dropout: float = 0.10,
    learnable_pe: bool = False,
    use_subsample: bool = True
):
    """
    Transformer time-series encoder for log-mel features.
    Input:  [B, T, F]
    Output: L2-normalized embedding of size EMB_DIM (compatible with your SimCLR/ProtoNet).
    """
    inp = Input(shape=input_shape)                      # [T, F]
    x = inp

    # Optional lightweight subsampling to reduce sequence length by 2x for stability/speed
    if use_subsample:
        x = layers.Conv1D(d_model, kernel_size=3, strides=2, padding="same", activation="relu")(x)  # [T/2, d_model]
    else:
        # Project feature dim -> d_model without changing T
        x = layers.Dense(d_model)(x)                    # [T, d_model]

    # Ensure channel dim is d_model
    if x.shape[-1] != d_model:
        x = layers.Dense(d_model)(x)

    # Positional encoding (sinusoidal by default; set learnable_pe=True to switch)
    x = PositionalEncoding(d_model, max_len=int(MAX_LENGTH), learnable=learnable_pe)(x)  # [T', d_model]
    x = layers.Dropout(dropout)(x)

    # Transformer blocks
    for _ in range(num_layers):
        x = TransformerBlock(d_model=d_model, num_heads=num_heads, mlp_dim=mlp_dim, drop=dropout)(x)

    # Global pooling over time
    x = layers.GlobalAveragePooling1D()(x)              # [B, d_model]

    # Projection to embedding dim + L2 norm
    z = layers.Dense(emb_dim)(x)                        # [B, EMB_DIM]
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
def _time_stretch_feat(x: np.ndarray, max_rate=0.15):
    # Approximate time-stretch in the feature domain (linear interpolation) to avoid relying on librosa.effects.time_stretch
    if max_rate <= 0: return x
    rate = 1.0 + rng.uniform(-max_rate, max_rate)  # [0.85, 1.15]
    T, F = x.shape
    new_T = max(1, int(T * rate))
    grid = np.linspace(0, T - 1, new_T)
    x_stretch = np.zeros((new_T, F), dtype=x.dtype)
    left = np.floor(grid).astype(int)
    right = np.clip(left + 1, 0, T - 1)
    w = grid - left
    x_stretch = (1 - w)[:, None] * x[left] + w[:, None] * x[right]
    # 回到原长（pad/trunc）
    if new_T >= T:
        return x_stretch[:T]
    else:
        out = np.zeros_like(x)
        out[:new_T] = x_stretch
        return out

def _mixup(a: np.ndarray, b: np.ndarray, alpha=0.2):
    lam = rng.beta(alpha, alpha)
    return lam * a + (1.0 - lam) * b

def make_ssl_batch(X, batch_size):
    idx = rng.choice(len(X), size=batch_size, replace=len(X)<batch_size)
    x0 = X[idx].copy()  # [B,T,F]

    # Light augmentations: random gain + time-stretch
    for i in range(x0.shape[0]):
        if rng.random() < 0.8:
            x0[i] = _rand_gain(x0[i])
        if rng.random() < 0.5:
            x0[i] = _time_stretch_feat(x0[i], max_rate=0.12)

    # 50% chance to apply Mixup with another sample in the same batch (convex combination)
    if rng.random() < 0.5:
        perm = rng.permutation(x0.shape[0])
        x0 = _mixup(x0, x0[perm], alpha=0.2)

    v1 = augment_view(x0)
    v2 = augment_view(x0)
    return v1.astype('float32'), v2.astype('float32')

@tf.function
def nt_xent_loss(z1, z2, temp=SSL_TEMP, var_w=0.05, cov_w=0.05, eps=1e-4):
    # InfoNCE
    z1 = tf.math.l2_normalize(z1, axis=-1)
    z2 = tf.math.l2_normalize(z2, axis=-1)
    z = tf.concat([z1, z2], axis=0)
    sim = tf.matmul(z, z, transpose_b=True)
    B = tf.shape(z1)[0]
    mask = tf.eye(2*B)
    sim = sim - 1e9 * mask
    logits = sim / temp
    labels = tf.concat([tf.range(B, 2*B), tf.range(0, B)], axis=0)
    info_nce = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    )

    # VICReg-like regularizers on projector outputs
    def _vicreg_terms(t):
        # Variance term: enforce per-dimension std ≥ 1 (avoid collapse)
        std = tf.sqrt(tf.math.reduce_variance(t, axis=0) + eps)  # [D]
        var_loss = tf.reduce_mean(tf.nn.relu(1.0 - std))

        # Decorrelation: penalize off-diagonal elements of the covariance across samples
        t = t - tf.reduce_mean(t, axis=0, keepdims=True)
        N = tf.cast(tf.shape(t)[0], t.dtype)
        cov = (tf.transpose(t) @ t) / (N - 1.0)  # [D,D]
        off_diag = cov - tf.linalg.diag(tf.linalg.diag_part(cov))
        cov_loss = tf.reduce_mean(tf.square(off_diag))
        return var_loss, cov_loss

    v1, c1 = _vicreg_terms(z1)
    v2, c2 = _vicreg_terms(z2)
    reg = var_w * (v1 + v2) + cov_w * (c1 + c2)

    return info_nce + reg

ssl_opt = tf.keras.optimizers.Adam(learning_rate=1e-3, global_clipnorm=1.0)
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
    mloss = float(np.mean(epoch_loss))
    # Monitor batch-level embedding stats: low variance -> warning
    v1, v2 = make_ssl_batch(X_train, SSL_BATCH)
    z_dbg = proj_head(encoder(v1, training=False), training=False)
    z_dbg = tf.math.l2_normalize(z_dbg, axis=-1).numpy()
    std_mean = float(z_dbg.std(axis=0).mean())
    cos = z_dbg @ z_dbg.T
    off_diag_mean = float((cos - np.eye(cos.shape[0])).mean())
    print(
        f"[SSL][{ep + 1:02d}/{SSL_EPOCHS}] loss={mloss:.4f} | emb.std̄={std_mean:.4f} | cos(offdiag)̄={off_diag_mean:.4f}")

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

        tau = 0.1
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(Yq, logits / tau, from_logits=True)
        )

    grads = tape.gradient(loss, enc.trainable_variables)
    opt.apply_gradients(zip(grads, enc.trainable_variables))

    preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
    acc = tf.reduce_mean(tf.cast(tf.equal(preds, Yq), tf.float32))
    return loss, acc

proto_opt = tf.keras.optimizers.Adam(learning_rate=1e-3, global_clipnorm=1.0)
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
print(" -", enc_path)
print(" -", proj_path)
print(" -", lin_path)
print(" -", label_path)
print(" -", config_path)
