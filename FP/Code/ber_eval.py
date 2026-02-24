"""
Project: Deep Learning for mMIMO-OFDM Channel Estimation
File: ber_eval.py
Author: Jeff K Chen
Date: 2025

Description:
    This script performs the Bit Error Rate (BER) evaluation to benchmark the 
    trained model against traditional estimation techniques.

Key Features:
    - Comparisons: Benchmarks DCCRN against Least Squares (LS) and LMMSE estimators.
    - Pipeline: Generates data -> Estimates Channel -> Zero-Forcing Detection -> Calculates BER.
    - Utilities: Supports automatic mixed-precision (FP16/FP32) inference and 
      generates BER vs. SNR plots using Matplotlib.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from mimo_ofdm_env import generate_mimo_channel, get_perfect_channel_freq, calculate_ber, lmmse_estimation, config

# ... (rest of the code)

import numpy as np
import torch
import matplotlib.pyplot as plt
from mimo_ofdm_env import generate_mimo_channel, get_perfect_channel_freq, calculate_ber, lmmse_estimation, config

def evaluate_ber_performance(model, device, test_batch_size=200, snr_points=None, plot_result=True, label_suffix=""):
    """
    執行完整的 BER 評估流程 (支援 FP16/FP32 自動偵測 + 即時打印)
    
    Args:
        label_suffix: 用於圖表標籤 (例如 " (FP16)") 以區分不同模型
    """
    
    if snr_points is None:
        snr_points = np.arange(0, 26, 5)

    # 自動偵測模型精度 (FP32 or FP16)
    try:
        model_dtype = next(model.parameters()).dtype
    except:
        model_dtype = torch.float32 # 預設

    dtype_str = "FP16" if model_dtype == torch.float16 else "FP32"
    print(f"🚀 Starting BER Evaluation [{dtype_str}] (Batch: {test_batch_size})...")
    
    ber_ls = []
    ber_lmmse = []
    ber_dccrn = []

    model.eval()

    for snr in snr_points:
        # 1. 生成數據
        h_time = generate_mimo_channel(test_batch_size)
        H_true = get_perfect_channel_freq(h_time) 
        H_true = H_true[..., np.newaxis]          

        sig_pwr = np.mean(np.abs(H_true)**2)
        noise_pwr = sig_pwr * 10**(-snr/10)
        noise = np.sqrt(noise_pwr/2) * (np.random.randn(*H_true.shape) + 1j*np.random.randn(*H_true.shape))
        H_ls = H_true + noise 

        # 2. 計算各方法 BER
        # LS & LMMSE (Baseline)
        ber_1 = calculate_ber(H_ls, H_true, snr)
        ber_ls.append(ber_1)

        H_lmmse = lmmse_estimation(H_ls, snr)
        ber_2 = calculate_ber(H_lmmse, H_true, snr)
        ber_lmmse.append(ber_2)

        # DCCRN
        x_real = np.real(H_ls).reshape(test_batch_size, -1, config.K, 1)
        x_imag = np.imag(H_ls).reshape(test_batch_size, -1, config.K, 1)
        dccrn_input = np.concatenate([x_real, x_imag], axis=1)
        
        # 轉換為對應的 dtype
        inp_tensor = torch.from_numpy(dccrn_input).to(device)
        inp_tensor = inp_tensor.to(model_dtype) 

        # 歸一化
        mean_val = torch.mean(inp_tensor)
        std_val = torch.std(inp_tensor) + 1e-6
        inp_norm = (inp_tensor - mean_val) / std_val

        with torch.no_grad():
            out_tensor = model(inp_norm)

        # 轉回 FP32 Numpy 計算 BER
        out_np = out_tensor.float().cpu().numpy()
        rx_tx = config.MIMO_Rx * config.MIMO_Tx
        
        h_pred_real = out_np[:, :rx_tx, :, :]
        h_pred_imag = out_np[:, rx_tx:, :, :]
        H_dccrn = (h_pred_real + 1j * h_pred_imag).reshape(test_batch_size, config.MIMO_Rx, config.MIMO_Tx, config.K, 1)

        ber_3 = calculate_ber(H_dccrn, H_true, snr)
        ber_dccrn.append(ber_3)
        
        # [NEW] 即時打印該 SNR 下的結果
        print(f"   SNR {snr:2d}dB | BER - LS: {ber_1:.5f} | LMMSE: {ber_2:.5f} | DCCRN: {ber_3:.5f}")

    # 3. 繪圖
    if plot_result:
        plt.figure(figsize=(10, 7))
        plt.semilogy(snr_points, ber_ls, 'b-o', label='LS')
        plt.semilogy(snr_points, ber_lmmse, 'g-s', label='LMMSE')
        plt.semilogy(snr_points, ber_dccrn, 'r-^', label=f'DCCRN {label_suffix}')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.legend()
        plt.show()

    return {
        "snr": snr_points,
        "ls": ber_ls,
        "lmmse": ber_lmmse,
        "dccrn": ber_dccrn
    }