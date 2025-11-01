# Animal Sound Recognition

## Table of Contents
- [1. Introduction](#1-introduction)
- [2. Literature Review](#2-literature-review)
- [3. Methods](#3-methods)
  - [3.1 Dataset and Task](#31-dataset-and-task)
  - [3.2 Feature Extraction](#32-feature-extraction)
  - [3.3 Model Architectures](#33-model-architectures)
  - [3.4 Data Augmentation & Regularization](#34-data-augmentation--regularization)
  - [3.5 Evaluation Metrics](#35-evaluation-metrics)
- [4. Evaluation](#4-evaluation)
  - [4.1 Quantitative Results](#41-quantitative-results)
  - [4.2 Discussion](#42-discussion)
  - [4.3 Practical Considerations](#43-practical-considerations)
- [5. Conclusion](#5-conclusion)
- [References](#references)

---

## 1. Introduction

This project targets animal sound recognition under small and imbalanced data constraints. Starting from a compact 1D-CNN baseline, we systematically examine how model capacity (AlexNet-Mel, Transformer) and augmentation strategies affect robustness. Early trials with naive wind-noise mixing revealed data collapse: when SNR and mixing policy are uncontrolled, the added noise dominates class cues and distorts the label distribution, causing severe overfitting and unstable validation curves.

To address this, we introduce a calibrated pipeline: standardized log-Mel features (16 kHz, 128 Mel), per-sample z-norm, SpecAugment, class weighting, and early stopping. Heavy models (AlexNet, Transformer) show high capacity but limited test gains on small data and longer training/infra cost. Our final system shows that moderate-capacity CNNs with calibrated augmentation deliver the best accuracy–stability trade-off, reaching 0.74 test accuracy while remaining reproducible and deployable on CPU.

---

## 2. Literature Review

Previous studies in environmental sound classification (ESC) have established log-Mel features and convolutional networks as reliable backbones for audio pattern recognition (Piczak, 2015; McFee et al., 2015). Deep architectures such as AlexNet (Krizhevsky et al., 2012) and PANNs (Kong et al., 2020) achieve high accuracy on large datasets like AudioSet (Gemmeke et al., 2017), but they depend heavily on abundant labeled data and extensive compute resources.

To enhance generalization in low-resource conditions, recent works employ data augmentation and self-supervised learning. SpecAugment (Park et al., 2019) applies time–frequency masking, while mixup (Zhang et al., 2018) interpolates examples to smooth decision boundaries. Contrastive methods such as SimCLR (Chen et al., 2020) and VICReg (Bardes et al., 2022) learn invariant representations without labels, and Prototypical Networks (Snell et al., 2017) improve few-shot adaptation.

Despite these advances, small bioacoustic datasets present unique challenges: class imbalance, environmental noise, and limited diversity make models sensitive to augmentation bias. Overly aggressive noise-mixing may reduce class separability—an effect rarely analyzed in prior work. Furthermore, lightweight yet expressive architectures remain underexplored for on-device animal recognition.

This project addresses these gaps by:
1. Comparing 1D-CNN, AlexNet, and Transformer architectures under equal training constraints;  
2. Quantifying how naive versus calibrated noise augmentation influences learning stability;  
3. Demonstrating that balanced augmentation plus regularization can achieve competitive accuracy without relying on large-scale pretraining.

---

## 3. Methods

### 3.1 Dataset and Task

The dataset “Animals_Sounds” contains short audio clips (16 kHz, mono) of animal calls from multiple species (e.g., Bear, Cat, Cow, Dog). Each subfolder represents a class. About 45 samples per class form the Training set, while a separate Test set ensures unbiased evaluation. Additionally, 40 wind noise recordings provide environmental interference for augmentation. The data is highly imbalanced, reflecting real-world field conditions.

### 3.2 Feature Extraction

All clips are resampled to 16 kHz, transformed into 128-bin log-Mel spectrograms using `n_fft=1024` and `hop_length=512`, then normalized (zero mean, unit variance). Each sequence is padded or truncated to 500 frames, ensuring consistent input length across the dataset.

### 3.3 Model Architectures

- **Baseline 1D-CNN**: Two Conv1D blocks (32, 64 filters) with batch normalization, max pooling, and dropout (0.3–0.4), followed by a dense classifier.  
- **AlexNet-Mel (2D-CNN)**: Adapts AlexNet to Mel-spectrogram “images”.  
- **Transformer-Mel**: Uses patchified spectrogram tokens with positional encoding and self-attention layers.  

All models optimize sparse categorical cross-entropy with Adam/AdamW, using class-weight rebalancing to counter imbalance.

### 3.4 Data Augmentation & Regularization

- Naive Noise-Mixing: Randomly adds wind noise (0–20 dB SNR). Effective only under mild conditions; too low SNR causes feature collapse.  
- Calibrated Noise-Mixing (Proposed): Limits SNR to 10–20 dB, enforces equal class coverage, and mixes noise only on training data.  
- SpecAugment: Applies 1–2 temporal and frequency masks.  
- Regularization: Dropout, early stopping, learning-rate scheduling, and class weighting.

### 3.5 Evaluation Metrics

The models are evaluated on Accuracy, Weighted F1, and per-class Precision/Recall. Weighted F1 is emphasized as it better represents imbalanced data behavior.

---

## 4. Evaluation

### 4.1 Quantitative Results

| Model Variant               | Test Accuracy | Weighted F1 | Observations                               |
|----------------------------|---------------|-------------|--------------------------------------------|
| 1D-CNN Baseline            | 0.45          | 0.44        | Stable but limited generalization          |
| + Naive Wind Mixing        | 0.47          | 0.47        | Slight improvement; prone to collapse      |
| AlexNet-Mel                | 0.55          | 0.54        | Overfits quickly; heavier computation      |
| Transformer-Mel            | 0.61          | 0.62        | Better training fit; weak test gain        |
| **Enhanced 1D-CNN (Ours)** | **0.74**      | **0.74**    | Best overall balance of accuracy and simplicity |

### 4.2 Discussion

The results confirm that complexity does not guarantee robustness. Both AlexNet and Transformer achieved high training accuracy but suffered from overfitting on the small dataset. The baseline 1D-CNN, when enhanced with calibrated noise and SpecAugment, provided the most reliable and interpretable results.

The observed data collapse during naive noise-mixing occurred because extreme SNR distortions overwhelmed class features, leading to low inter-class variance in latent space. Adjusting SNR bounds and per-class augmentation quotas restored discriminability and improved F1 scores by 5–6%.

Further ablation studies demonstrated that removing SpecAugment or class weighting degraded minority-class recall significantly. This underlines the importance of both feature-level and loss-level regularization when dealing with small-scale imbalanced datasets.

### 4.3 Practical Considerations

While larger models require longer training and higher computational resources, the enhanced 1D-CNN maintains low latency and minimal memory usage—suitable for edge deployment or educational demonstration systems. Moreover, its reproducibility (via fixed configuration files and consistent preprocessing) makes it a dependable benchmark for future extensions.

---

## 5. Conclusion

This study systematically analyzed animal sound recognition under data scarcity and imbalance. The experiments showed that moderate-capacity CNNs, combined with carefully tuned augmentation, deliver the best trade-off between generalization and complexity. Naive noise addition can degrade performance through feature homogenization, whereas calibrated SNR control and SpecAugment enhance robustness.

The project contributes a clear framework for future small-data bioacoustic tasks—highlighting that “simpler but well-regularized” models often outperform deep architectures when training data are limited.

Future work will explore semi-supervised pretraining, domain adaptation, and test-time augmentation to further improve model stability across unseen acoustic environments.

---

## References

1. Piczak, K. J. (2015). ESC: Dataset for Environmental Sound Classification.  
2. McFee, B., et al. (2015). Librosa: Audio and Music Signal Analysis in Python.  
3. Park, D. S., et al. (2019). SpecAugment: A Simple Data Augmentation Method for ASR.  
4. Kong, Q., et al. (2020). PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition.  
5. Gemmeke, J. F., et al. (2017). AudioSet: An Ontology and Human-Labeled Dataset for Audio Events.  
6. Chen, T., et al. (2020). SimCLR: A Simple Framework for Contrastive Learning of Visual Representations.  
7. Snell, J., et al. (2017). Prototypical Networks for Few-Shot Learning.  
8. Bardes, A., et al. (2022). VICReg: Variance-Invariance-Covariance Regularization for SSL.  
9. Gong, Y., et al. (2021). AST: Audio Spectrogram Transformer.  
10. Krizhevsky, A., et al. (2012). ImageNet Classification with Deep Convolutional Neural Networks (AlexNet).  
11. Sandler, M., et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks.  
12. Lin, T.-Y., et al. (2017). Focal Loss for Dense Object Detection.  
13. Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization.  
14. Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT).
