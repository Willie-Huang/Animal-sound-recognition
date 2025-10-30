# Animal Sound Recognition

> A progressively enhanced, noise-tolerant pipeline that classifies animal sounds from short audio clips using log-Mel features and a family of CNN and Transformer models. The project begins with a **1D-CNN baseline (~45% accuracy)** and advances to **Transformer and SimCLR-based architectures (~74% accuracy)**. Supports microphone inference with dataset fallback when PyAudio is unavailable.

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
This repository provides an end-to-end **animal sound recognition system**.  
It transforms raw `.wav` files into **128-dimensional log-Mel spectrograms**, normalizes each sample, and feeds them into a series of deep models ranging from a **1D convolutional baseline** to **Transformer encoders** with **SimCLR self-supervised pretraining** and **ProtoNet fine-tuning**.  
The pipeline supports training, model export (`.keras`), and real-time microphone inference with automatic fallback if PyAudio is unavailable.

---

## 2. Objectives
- **Establish a Baseline**: Implement a small and efficient **1D-CNN** model using log-Mel features.  
- **Enhance Robustness**: Improve classification under **wind and environmental noise** through SNR-controlled augmentation.  
- **Representation Learning**: Incorporate **SimCLR** and **Prototypical Networks** to improve generalization in few-shot conditions.  
- **Model Evaluation**: Compare 1D-CNN, AlexNet, MobileNetV2, and Transformer-based models on small, imbalanced datasets.  
- **Ease of Use**: Enable one-command training and inference with reproducible outputs in the `model/` directory.  

---

## 3. Datasets

Animals_Sounds/
├─ Training/
│ ├─ Bear/
│ ├─ Cat/
│ ├─ Cow/
│ ├─ Dog/
│ └─ ...
├─ Test/
│ ├─ Bear/
│ ├─ Cat/
│ ├─ Cow/
│ ├─ Dog/
│ └─ ...
└─ Noise/
└─ wind/

> Each subfolder represents one class label.  
> Wind noise clips are used for **training augmentation only** (not for testing).

---

## 4. Methods

### Feature Pipeline
- **Resampling**: `sr = 16000`  
- **Mel spectrogram**: `n_mels = 128`, `n_fft = 1024`, `hop_length = 512`  
- **Post-processing**: Power→dB conversion, per-sample z-normalization  
- **Sequence shaping**: `pad_sequences(maxlen=500, padding='post', truncating='post')`  

### Model Architecture
**Baseline 1D-CNN**
Conv1D(32, k=3) → BatchNorm → MaxPool → Dropout(0.3)
Conv1D(64, k=3) → BatchNorm → MaxPool → Dropout(0.3)
Flatten → Dense(128) → Dropout(0.4) → Dense(num_classes, softmax)

**Variants**
- **MixNoise_1D_CNN**: baseline + naive wind-mixing (0–20 dB SNR).  
- **AlexNet-Mel (2D CNN)**: large image-style architecture on spectrograms.  
- **Transformer + SimCLR + ProtoNet**: uses self-supervised contrastive learning followed by few-shot fine-tuning.  
- **Enhanced_1D_CNN (Best)**: baseline + calibrated noise augmentation, SpecAugment, class weights, and higher dropout.

### Default Hyperparameters
```python
EPOCHS = 30
BATCH_SIZE = 32
SR = 16000
N_MELS = 128
N_FFT = 1024
HOP = 512
MAX_LENGTH = 500
TEST_SIZE = 0.2
RANDOM_STATE = 42

5. Installation & Training
# 1) Install dependencies
pip install -r requirements.txt
export KERAS_BACKEND=tensorflow

# 2) Train (choose model variant)
python baseline.py
python mix_noise.py
python "train AlexNet.py"
python "enhance transformer.py"
python enhance.py

After completion, model/ will contain:

animal_sound_model.keras

label_encoder.pkl

config.json

Evaluation CSVs (classification report, confusion matrix)

## 6.Inference
File mode

python predict.py --file /path/to/audio.wav


Microphone mode

python prediction.py
# or if capitalized:
python Prediction.py


If microphone permissions are unavailable, the program automatically falls back to a dataset audio sample.
On macOS, install PortAudio/PyAudio for live recording.

## 7. Repository Structure
Sound-Animal-Recognition/
├─ Animals_Sounds/
│  ├─ Training/
│  ├─ Test/
│  └─ Noise/
├─ charts/
│  ├─ chart1_precision_per_class.png
│  ├─ chart2_recall_per_class.png
│  ├─ chart3_f1_per_class.png
│  ├─ chart4_test_accuracy.png
│  └─ chart5_macro_vs_weighted_f1.png
├─ baseline.py
├─ mix_noise.py
├─ enhance.py
├─ enhance transformer.py
├─ train AlexNet.py
├─ Prediction.py
├─ requirements.txt
└─ model/
## 8. Planned Experiments (Roadmap)

Augmentation: SpecAugment, mixup, gain jitter, and pitch shift.

Architecture: CRNN (Conv+BiGRU), Transformer-light, MobileNetV2 embedding head.

Loss and Fairness: Focal loss, label smoothing, and class-balanced sampling.

Cross-domain Testing: Train on one noise domain, test on another.

Deployment: Quantized TFLite/CoreML model for mobile inference.

## 9. Evaluation

## 10. Risks, Limitations & Ethics
Domain shift: Real-world noise differs from training data; mitigated through cross-domain evaluation.

Imbalanced data: Small sample sizes per class require reweighting or resampling.

Overfitting risk: Use dropout, data augmentation, and early stopping.

Ethical use: Predictions are probabilistic; not a substitute for expert bioacoustic analysis.

Privacy: Ensure recorded samples comply with local data protection laws.

## References
1. Piczak, K. J. (2015). ESC: Dataset for Environmental Sound Classification.
2. McFee, B., et al. (2015). librosa: Audio and Music Signal Analysis in Python.
3. Park, D. S., et al. (2019). SpecAugment: A Simple Data Augmentation Method for ASR.
4. Kong, Q., et al. (2020). PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition.
5. Gemmeke, J. F., et al. (2017). AudioSet: An Ontology and Human-Labeled Dataset for Audio Events.
6. Chen, T., et al. (2020). SimCLR: A Simple Framework for Contrastive Learning of Visual Representations.
7. Snell, J., et al. (2017). Prototypical Networks for Few-shot Learning.
8. Bardes, A., et al. (2022). VICReg: Variance-Invariance-Covariance Regularization for SSL.
9. Gong, Y., et al. (2021). AST: Audio Spectrogram Transformer.
10. Krizhevsky, A., et al. (2012). ImageNet Classification with Deep Convolutional Neural Networks (AlexNet).
11. Sandler, M., et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks.
12. Lin, T.-Y., et al. (2017). Focal Loss for Dense Object Detection.
13. Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization.
14. Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT).
