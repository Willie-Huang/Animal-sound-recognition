# Animal Sound Recognition 

> A noise-robust, lightweight pipeline for classifying animal sounds with log-Mel spectrograms and CNN/CRNN backbones, plus transfer learning from audio foundation models.

## 1. Overview
This project builds an end-to-end Python pipeline to recognize animal sounds from short audio clips. A baseline model uses log-Mel spectrograms with a compact CNN; stretch goals explore CRNNs and transfer learning (e.g., PANNs, YAMNet). Passive acoustic monitoring enables scalable wildlife surveys and long-term ecosystem tracking (Piczak, 2015; Salamon & Bello, 2014). We focus on practical robustness under field noise with principled augmentations and cross-dataset evaluation (McFee et al., 2015; Park et al., 2019; Kong et al., 2020).

## 2. Objectives
1. **Baseline**: Train a CNN on ESC-50 animal classes using log-Mel inputs.  
2. **Noise robustness**: Add SpecAugment, time shift, and mixup to handle field noise.  
3. **Transfer learning**: Evaluate PANNs/YAMNet embeddings and fine-tuning for few-shot gains.  
4. **Evaluation**: Report macro-F1, ROC-AUC; provide confusion matrices and ablations.  
5. **Edge demo (stretch)**: Export to TFLite/ONNX for on-device inference.

## 3. Datasets
- **ESC-50 (animal subset)** — curated environmental sounds across 50 classes (Piczak, 2015).  
- **UrbanSound8K** — urban sounds incl. animal-related classes; good for robustness baselines (Salamon & Bello, 2014).  
- **Bird-centric (stretch)** — BirdCLEF / Cornell birdcall corpora for fine-grained species recognition.  
> Tooling: `librosa` for audio I/O and feature extraction (McFee et al., 2015).

## 4. Methods
### 4.1 Pre-processing
- Resample to 16 kHz mono → pre-emphasis → framing → **log-Mel spectrogram** (64–128 mels).  
- Per-file standardization; optional VAD/energy gating for silence trimming.

### 4.2 Models
- **CNN-Small (baseline)**: 3–5 conv blocks → global pooling → softmax.  
- **CRNN (stretch)**: 2D CNN feature extractor + GRU for temporal context.  
- **Foundation models**:  
  - **PANNs** (AudioSet-pretrained; CNN14/ResNet variants) for embeddings / fine-tune.  
  - **YAMNet** (MobileNet-v1 trunk; 521 AudioSet classes) as embedding front-end.

### 4.3 Data Augmentation
SpecAugment (time/freq masking), random time shift, gain jitter, and optional background mixing (Park et al., 2019).


## 5. Repository Structure（仓库结构）
animal-sound-recognition/
├─ src/
│ ├─ datasets.py # ESC-50/UrbanSound8K loaders (librosa)
│ ├─ models.py # CNN/CRNN + PANNs/YAMNet heads
│ ├─ train.py # training loop, logging, checkpoints
│ ├─ eval.py # metrics & confusion matrices
│ └─ utils_audio.py # feature extraction (log-Mel, SpecAug)
├─ notebooks/ # EDA & error analysis
├─ data/ # (gitignored)
├─ results/ # metrics, plots, audio demos
├─ docs/ # GitHub Pages site
├─ requirements.txt
└─ README.md

## 6. Planned Experiments
**E0. Data hygiene & baselines**  
- Curate ESC-50 animal subset; stratified train/val/test (use official folds when applicable).  
- Train **CNN-Small** on log-Mel inputs; record macro-F1 / ROC-AUC.

**E1. Augmentation ablations**  
- Compare **No-Aug** vs **SpecAugment** vs **SpecAugment + time-shift + mixup**.  
- Hypothesis H1: SpecAugment improves macro-F1, especially for minority animal classes (Park et al., 2019).

**E2. Model family comparison**  
- CNN-Small vs **CRNN**.  
- Hypothesis H2: CRNN improves per-class recall for temporally variable calls.

**E3. Transfer learning**  
- **PANNs embeddings** (frozen) + linear probe; then partial fine-tuning.  
- **YAMNet embeddings** + linear probe / adapter.  
- Hypothesis H3: Transfer models yield higher macro-F1 in low-data regimes (Kong et al., 2020).

**E4. Robustness & domain shift**  
- Add **UrbanSound8K** animal-related clips as out-of-domain noise.  
- Train on ESC-50; test on held-out noisy splits (cross-dataset).  
- Hypothesis H4: Augmented training narrows the performance gap under domain shift.

**E5. Few-shot stress test (stretch)**  
- Downsample per-class training to K ∈ {1, 5, 10} shots; compare baselines vs transfer.  
- Measure accuracy vs K-shots to quantify data efficiency.

**E6. Inference efficiency (stretch)**  
- Measure latency and model size; evaluate ONNX/TFLite exports for edge feasibility.

## 7. Evaluation
- **Primary metrics**: **Macro-F1** (class imbalance-aware), **macro ROC-AUC**.  
- **Secondary**: mAP, per-class F1/Recall, confusion matrix, PR curves.  
- **Validation protocol**:  
  - Use official **ESC-50 5-fold CV** where applicable to avoid leakage (Piczak, 2015).  
  - Report **mean ± std** across folds and random seeds.  
- **Statistical testing**: Paired t-test or Wilcoxon signed-rank on per-fold metrics for key comparisons (e.g., No-Aug vs SpecAugment; CNN vs CRNN; baseline vs PANNs/YAMNet).  
- **Robustness checks**: Evaluate under SNR degradation (additive noise at {20, 10, 0 dB}); report **degradation curves** of macro-F1 vs SNR.  
- **Error analysis**:  
  - Top-k confusion pairs (e.g., similar timbres).  
  - Saliency/Grad-CAM on spectrograms to verify model focus on harmonics/formants vs background.  
- **Model efficiency**: Params, FLOPs, latency (CPU-only and GPU), and model size; summarize in a single comparison table.

---

## 8. Risks, Limitations & Ethics
- **Class imbalance / noise**: weighted loss, re-sampling, stronger augmentation.  
- **Domain shift**: curated vs field recordings; apply background mixing, cross-dataset tests.  
- **Licensing & ethics**: respect dataset licenses; avoid misuse for hunting/harassment; anonymize sensitive location metadata.

---

## References
- Gemmeke, J. F., Ellis, D. P. W., Freedman, D., Jansen, A., Lawrence, W., Moore, R. C., & Plakal, M. (2017). Audio Set: An ontology and human-labeled dataset for audio events. *IEEE ICASSP 2017*, 776–780.  
- Kong, Q., Cao, Y., Iqbal, T., Wang, Y., Plumbley, M. D., & Wang, W. (2020). PANNs: Large-scale pretrained audio neural networks for audio pattern recognition. *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 28, 2880–2894.  
- McFee, B., Raffel, C., Liang, D., Ellis, D. P. W., McVicar, M., Battenberg, E., & Nieto, O. (2015). librosa: Audio and music signal analysis in Python. In *Proceedings of the 14th Python in Science Conference* (pp. 18–24).  
- Park, D. S., Chan, W., Zhang, Y., Chiu, C.-C., Zoph, B., Cubuk, E. D., & Le, Q. V. (2019). SpecAugment: A simple data augmentation method for automatic speech recognition. *INTERSPEECH 2019*, 2613–2617.  
- Piczak, K. J. (2015). ESC: Dataset for environmental sound classification. In *Proceedings of the 23rd ACM International Conference on Multimedia* (pp. 1015–1018).  
- Salamon, J., & Bello, J. P. (2014). A dataset and taxonomy for urban sound research. In *Proceedings of the 22nd ACM International Conference on Multimedia* (pp. 1041–1044).
