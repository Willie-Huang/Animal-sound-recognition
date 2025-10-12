# Animal Sound Recognition

> A lightweight, noise-tolerant pipeline that classifies animal sounds from short audio clips using log-Mel features and a compact 1D-CNN. Supports microphone inference with automatic dataset fallback when PyAudio is unavailable.

---

## Table of Contents
- [1. Overview](#1-overview)
- [2. Objectives](#2-objectives)
- [3. Datasets](#3-datasets)
- [4. Methods](#4-methods)
  - [Feature Pipeline](#feature-pipeline)
  - [Model Architecture](#model-architecture)
  - [Default Hyperparameters](#default-hyperparameters)
- [5. Installation & Training](#5-installation--training)
- [6. Inference](#6-inference)
- [7. Repository Structure](#7-repository-structure)
- [8. Planned Experiments (Roadmap)](#8-planned-experiments-roadmap)
- [9. Evaluation](#9-evaluation)
- [10. Risks, Limitations & Ethics](#10-risks-limitations--ethics)
- [References](#references)

---

## 1. Overview
This project provides an end-to-end sound recognition pipeline: it converts `.wav`/`.ogg` audio files to **16 kHz**, extracts **128-dimensional log-Mel** sequences, pads/truncates to a fixed length (**500 frames**), and feeds them into a small **Conv1D** model for classification. Training saves the model (`.keras`), label encoder (`.pkl`), and configuration (`config.json`). Inference strictly reuses the same features and configuration to ensure reproducibility.

---

## 2. Objectives
- **Baseline**: An ESC-style classifier based on **128-Mel + Conv1D**, CPU-friendly and easy to train.  
- **Robust inference**: Prioritize **5 s** of microphone recording; if PyAudio is missing or permissions are unavailable, automatically fall back to the first audio in the dataset for demonstration.  
- **Reproducibility**: Training and inference share a standardized pipeline, with artifacts persisted to `model/`.

---

## 3. Datasets

Animals_Sounds/
├─ Bear/ bear_001.wav …
├─ Cat/
├─ Cow/
├─ Dog/
└─ …

> Each subfolder name is treated as a class label.

---

## 4. Methods

### Feature Pipeline
- **Resampling**: `sr = 16000`  
- **Mel spectrogram**: `n_mels = 128`, `n_fft = 1024`, `hop_length = 512`  
- **Post-processing**: dB conversion → per-sample, per-band normalization (subtract mean / divide std)  
- **Sequence shaping**: `pad_sequences(maxlen=500, padding='post', truncating='post')`

### Model Architecture (Keras Sequential, 1D-CNN)

Conv1D(32, k=3) → BatchNorm → MaxPool → Dropout(0.3)
Conv1D(64, k=3) → BatchNorm → MaxPool → Dropout(0.3)
Flatten → Dense(128) → Dropout(0.4) → Dense(num_classes, softmax)

- **Loss**: `sparse_categorical_crossentropy`  
- **Optimizer**: `adam`

### Default Hyperparameters

EPOCHS = 30
BATCH_SIZE = 32
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_LENGTH = 500
N_MELS = 128


---

## 5. Installation & Training
```bash
# 1) Install dependencies
pip install -r requirements.txt
# Optional: choose TF backend for Keras
export KERAS_BACKEND=tensorflow

# 2) Train
python train.py


After completion, model/ will contain:

animal_sound_model.keras
label_encoder.pkl
config.json

6. Inference
File mode
python predict.py --file /path/to/audio.wav

Microphone mode (5 seconds; auto-fallback if PyAudio/permissions are missing)
python prediction.py
# Note: If your script is named `Prediction.py` (capital P), run:
# python Prediction.py


If animal_pictures/<Label>.jpeg exists, the image is displayed alongside the prediction.

macOS tip: For microphone recording, install PortAudio/PyAudio. If not installed or permission is denied, the command will automatically fall back to a dataset sample without error.

7. Repository Structure
Sound-Animal-Recognition/
├─ Animals_Sounds/        # Dataset (subfolder = category)
├─ animal_pictures/       # Optional: <Label>.jpeg per class (Bear.jpeg, Cat.jpeg, ...)
├─ model/                 # Generated after training
│  ├─ animal_sound_model.keras
│  ├─ label_encoder.pkl
│  └─ config.json
├─ train.py               # Training script
├─ Prediction.py          # Inference script (microphone/file + fallback)
├─ requirements.txt
└─ temp.wav               # Temp mic recording file (runtime-generated)

8. Planned Experiments (Roadmap)

Data augmentation: time shift, mixup, SpecAugment (time/freq mask)

Model extension: CRNN (Conv + BiGRU), lighter CNN variants

Transfer learning: PANNs / YAMNet as embeddings + linear/MLP heads

Cross-dataset generalization: train-A / test-B robustness evaluation

Edge metrics: CPU latency, memory usage, model size comparison

9. Evaluation

Test Accuracy, Top-k accuracy

Confusion matrix

Per-class Precision/Recall

ROC-AUC (OvR)

Error analysis on easily confused class pairs

10. Risks, Limitations & Ethics

Domain shift: Field noise may not align with training distribution → adopt augmentation and cross-domain evaluation.

Class imbalance: Use stratified sampling / loss reweighting.

Responsible use: Outputs are probabilistic and do not replace bioacoustic expert judgment.

Privacy: Obtain consent for recordings to avoid uploading restricted material.

References

Piczak, K. J. (2015). ESC: Dataset for Environmental Sound Classification.

McFee, B., et al. (2015). librosa: Audio and Music Signal Analysis in Python.

Park, D. S., et al. (2019). SpecAugment.

Kong, Q., et al. (2020). PANNs: Large-Scale Pretrained Audio Neural Networks.

YAMNet / AudioSet (Google Research).
