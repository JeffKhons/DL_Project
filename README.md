# 📡 Deep Learning for MIMO-OFDM Channel Estimation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

> **Fast, Accurate, and Lightweight.** A comprehensive study on replacing traditional LMMSE channel estimation with Deep Learning (DCCRN) in 5G MIMO-OFDM systems.

## 📖 Abstract
Channel estimation is a critical component in 5G/6G communications. Traditional **LMMSE (Linear Minimum Mean Square Error)** algorithms, while accurate, suffer from high computational complexity O(K^3), making them impractical for real-time massive MIMO systems.

This project proposes a **4-Layer Deep Complex Convolutional Recurrent Network (DCCRN)** to estimate channel state information (CSI). We further optimize the model using **FP16 Quantization** and **Iterative Unstructured Pruning (85% Sparsity)**.

**Key Achievements:**
* **Accuracy**: Matches LMMSE performance (-22dB NMSE) with no error floor.
* **Speed**: Achieves **30x-50x speedup** in throughput mode (Batch Processing) compared to CPU-based LMMSE.
* **Efficiency**: Reduces model storage size by **~2.33x** (Pruning + Compression) with negligible accuracy loss.

---

## 🏗️ System Model & Methodology

### 1. 5G MIMO-OFDM Environment
* **Configuration**: 8x8 MIMO, 512 Subcarriers, QPSK Modulation.
* **Channel Model**: Rayleigh Fading with Doppler shift (simulating urban mobility).
* **Pilot Config**: Comb-type pilots (Spacing=8).

### 2. Model Architecture: DCCRN
* **Backbone**: U-Net style architecture with **Complex Convolutions**.
* **Input**: Noisy LS (Least Squares) estimates.
* **Output**: Denoised Channel estimates.

### 3. Optimization Strategy
* **Curriculum Learning**: Mixed SNR training (5dB to 30dB) for robustness.
* **Scheduler**: Cosine Annealing LR for precise convergence.
* **Model Compression**:
    1.  **Quantization**: FP32 $\to$ FP16 (Half Precision).
    2.  **Pruning**: Iterative Global Unstructured Pruning ($50\% \to 70\% \to 85\%$).

---

## 📊 Experimental Results

### 1. BER Performance (Accuracy)
We compared DCCRN against LS and LMMSE baselines.
* **LS**: Poor performance due to noise.
* **LMMSE**: High accuracy but computationally heavy.
* **DCCRN (Pruned+FP16)**: **Matches LMMSE accuracy** while being extremely lightweight.

### 2. Throughput & Latency (Speed)
We tested the "Average Latency Per User" under batch processing (Batch=4).
* **LMMSE (CPU)**: ~55~65ms per user (Sequential bottleneck).
* **DCCRN (GPU)**: **< 1ms per user** (Parallelization advantage).

### 3. Compression Analysis (Storage)
By applying **85% Unstructured Pruning** and saving with standard compression (ZIP), we proved effective storage reduction.

| Model Type | Raw Size (.pth) | Compressed Size (.zip) | Ratio |
| :--- | :--- | :--- | :--- |
| **FP16 Only** | ~X MB | ~Y MB | 1.0x |
| **Pruned 85% + FP16** | ~X MB | **~Z MB** | **~2.33x** |

---

## 🔧 Training Configuration

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Epochs** | 8,000 | 20 Stages of curriculum learning |
| **Batch Size** | 32 | Mixed SNR batches |
| **Optimizer** | AdamW | Weight Decay = 5e-4 |
| **LR Schedule** | Cosine Annealing | 5e-4 $\to$ 1e-7 |
| **Loss** | MSE | Mean Squared Error |

---

## 📂 Project Structure

```bash
├── mimo_ofdm_env.py     # 5G System simulation (Channel generation, BER calc)
├── dccrn_module.py      # DCCRN Model Architecture (Complex Conv layers)
├── ber_eval.py          # Evaluation utilities (LS, LMMSE, Model inference)
├── train.ipynb          # Main notebook for training and analysis
├── dccrn_mimo_8x8.pth   # Pre-trained weights
└── README.md            # This file


## 🚀 如何執行 (How to Run)

本專案建議在 Google Colab 或具備 NVIDIA GPU 的環境下執行，以獲得最佳的運算加速效果。

### 1. 環境準備 (Prerequisites)

請確保您的環境已安裝以下 Python 套件：
```bash
pip install torch numpy matplotlib tqdm

### 2. 執行 (Run)

Please use /code/main.ipynb to run.






