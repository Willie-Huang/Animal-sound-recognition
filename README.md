# Animal Sound Recognition 

> A lightweight, noise-tolerant pipeline that classifies animal sounds from short audio clips using log-Mel features and a compact 1D-CNN. Supports microphone inference with automatic dataset fallback when PyAudio is unavailable.

## 1. Overview
This project provides an end-to-end sound recognition pipeline: It converts .wav/.ogg audio files to 16 kHz, extracts 128-dimensional log-Mel spectrum sequences, pads/truncates them at a fixed time step (500 frames), and feeds them into a small Conv1D model for classification. Training saves the model (.keras), label encoder (.pkl), and configuration (config.json). Inference strictly reuses the same set of features and configuration to ensure reproducibility.

## 2. Objectives
Baseline: An ESC-style classifier based on 128-Mel + Conv1D, CPU-friendly and easy to train.
Robust Inference: Prioritizes 5s of microphone recording; if PyAudio is missing or permissions are unavailable, it automatically falls back to the first audio in the dataset for demonstration.
Reproducibility: Training and inference share a standardized, hyper-participatory approach, with artifacts persisted to model/.

## 3. Datasets
Animals_Sounds/
├─ Bear/      bear_001.wav …
├─ Cat/
├─ Cow/
├─ Dog/
└─ …

## 4. Methods
Feature pipeline (training and inference are consistent)

Resampling: SR=16000

Mel spectrum: n_mels=128, n_fft=1024, hop=512

After dB conversion, normalize each sample and band (subtract mean/divide standard deviation)

pad_sequences(maxlen=500, padding='post', truncating='post')

Model architecture (Keras Sequential, 1D-CNN)

Conv1D(32, k=3) → BN → MaxPool → Dropout(0.3)

Conv1D(64, k=3) → BN → MaxPool → Dropout(0.3)

Flatten → Dense(128) + Dropout(0.4) → Dense(num_classes, softmax)

Loss: sparse_categorical_crossentropy; Optimizer: adam

Default hyperparameters: EPOCHS=30, BATCH_SIZE=32, TEST_SIZE=0.2, RANDOM_STATE=42, MAX_LENGTH=500, N_MELS=128

Installation and Training

pip install -r requirements.txt
# Optional: export KERAS_BACKEND=tensorflow
python train.py
After completion, the following will be generated in model/ :
animal_sound_model.keras, label_encoder.pkl, config.json
Inference (two modes)

# File mode
python predict.py --file /path/to/audio.wav

# Microphone mode (5 seconds). If PyAudio/permissions are missing, the command will automatically fall back to the dataset sample.
python prediction.py
If animal_pictures/<Label>.jpeg exists, the image will be displayed.
macOS Tip: To enable microphone recording, install PortAudio/PyAudio first; otherwise, running the command directly will automatically fall back without an error.

## 5. Repository Structure
Sound-Animal-Recognition/
├─ Animals_Sounds/ # Dataset (subfolder = category)
├─ animal_pictures/ # Optional: Images with the same name as the category (Bear.jpeg, Cat.jpeg, ...)
├─ model/ # Model and configuration generated after training
│ ├─ animal_sound_model.keras
│ ├─ label_encoder.pkl
│ └─ config.json
├─ train.py # Training script
├─ Prediction.py # Inference script (microphone/file + fallback)
├─ requirements.txt
└─ temp.wav # Temporary microphone recording file (generated at runtime)

## 6. Planned Experiments
Data augmentation: time shift, mixup, SpecAugment (time/freq mask)
Model extension: CRNN (Conv + BiGRU), lighter CNN variants
Transfer learning: PANNs/YAMNet as embeddings + linear/MLP heads
Cross-dataset generalization: train-A/test-B robustness evaluation
Edge metrics: CPU latency, memory usage, model size comparison

## 7. Evaluation
Test Accuracy, confusion matrix, per-class precision/recall, ROC-AUC (OvR), Top-k accuracy, and analysis of easily confused class pairs.

## 8. Risks, Limitations & Ethics
Domain shift: Field noise may not align with the training distribution → Enhancement and cross-domain evaluation are required.
Class imbalance: Stratified sampling/loss reweighting can be used.
Responsible use: Output is a probabilistic prediction and does not replace bioacoustic expert judgment.
Privacy compliance: Requires consent for recordings to avoid uploading restricted material.

## References
1. Piczak, K. J. (2015). ESC: Dataset for Environmental Sound Classification.
2. McFee, B., et al. (2015). librosa: Audio and Music Signal Analysis in Python.
3. Park, D. S., et al. (2019). SpecAugment.
4. Kong, Q., et al. (2020). PANNs: Large-Scale Pretrained Audio Neural Networks.
5. YAMNet / AudioSet (Google Research).
