# Animal Sound Recognition 

> A noise-robust, lightweight pipeline for classifying animal sounds with log-Mel spectrograms and CNN/CRNN backbones, plus transfer learning from audio foundation models.

## 1. Overview
This project builds an end-to-end Python pipeline to recognize animal sounds from short audio clips. A baseline model uses log-Mel spectrograms with a compact CNN; stretch goals explore CRNNs and transfer learning (e.g., PANNs, YAMNet). Target applications include biodiversity monitoring and on-device eco-acoustics.

**Motivation.** Passive acoustic monitoring enables scalable wildlife surveys and long-term ecosystem tracking (Piczak, 2015; Salamon & Bello, 2014).

---

## 2. Objectives
1. **Baseline**: Train a CNN on public datasets (e.g., ESC-50 animal classes) using log-Mel inputs.  
2. **Noise robustness**: Add SpecAugment, time shift, and mixup to handle field noise (Park et al., 2019).  
3. **Transfer learning**: Evaluate PANNs/YAMNet embeddings and fine-tuning (Kong et al., 2020).  
4. **Evaluation**: Report macro-F1, ROC-AUC; provide confusion matrices and ablations.  
5. **Edge demo (stretch)**: Export to TFLite/ONNX for on-device inference.

---

## 3. Datasets (candidate)
- **ESC-50 (animal subset)** — curated environmental sounds across 50 classes (Piczak, 2015).  
- **UrbanSound8K** — urban sounds including animal-related classes; useful for robustness (Salamon & Bello, 2014).  
- **Bird-centric (stretch)** — BirdCLEF / Cornell birdcall corpora for fine-grained species recognition.

> Tooling: `librosa` for audio I/O and feature extraction (McFee et al., 2015).

---

## 4. Methods

### 4.1 Pre-processing
- 16 kHz mono resampling → pre-emphasis → framing → **log-Mel spectrogram** (64–128 mels).  
- Per-file standardization; optional VAD/energy gating for silent regions.

### 4.2 Models
- **CNN-Small (baseline)**: 3–5 conv blocks → global pooling → softmax.  
- **CRNN (stretch)**: 2D CNN feature extractor + GRU for temporal context.  
- **Foundation models**:  
  - **PANNs** (AudioSet-pretrained; CNN14/ResNet variants).  
  - **YAMNet** (MobileNet-v1 trunk; 521 AudioSet classes).

### 4.3 Data Augmentation
SpecAugment (time/freq masking), random time shift, gain jitter, and optional background mixing.

---

## 5. Metrics & Protocol
- **Primary**: Macro-F1, macro ROC-AUC.  
- **Secondary**: mAP, per-class F1, confusion matrix.  
- **Splits**: Stratified train/val/test; preserve official folds for ESC-50 to avoid leakage.

---

## 6. Quick Start

```bash
# 1) Environment
conda create -n animal-audio python=3.10 -y
conda activate animal-audio
pip install torch torchaudio librosa scikit-learn matplotlib numpy tqdm

# 2) Data (example: ESC-50)
# Download and place under: data/ESC-50/{audio/*, meta/*}

# 3) Train baseline
python src/train.py --dataset esc50 --epochs 30 --model cnn_small --specaug true

# 4) Evaluate
python src/eval.py --ckpt runs/best.pt
