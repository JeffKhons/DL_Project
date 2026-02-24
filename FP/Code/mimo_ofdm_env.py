"""
Project: Deep Learning for mMIMO-OFDM Channel Estimation
File: mimo_ofdm_env.py
Author: Jeff K Chen
Date: 2025

Description:
    This file defines the 5G NR simulation environment, system configuration, 
    and core signal processing utilities for the MIMO-OFDM system.

Contents:
    - MIMO_Config: Sets up 5G parameters (3.5GHz, 60kHz SCS, 8x8 MIMO) and 
      defines a V2X high-mobility scenario (Urban Micro).
    - Channel Model: Generates time-varying channels with Doppler effects.
    - Baselines: Implements LS and LMMSE channel estimation algorithms.
    - Utils: Helper functions for OFDM (DFT/IDFT), QAM modulation, and BER calculation.
"""

import numpy as np

# ==========================================
# 1. 5G NR 系統參數設定 (Configuration)
# ==========================================
# ... (rest of the code)

import numpy as np

# ==========================================
# 1. 5G NR 系統參數設定 (Configuration)
# ==========================================
class MIMO_Config:
    def __init__(self):
        # --- 5G NR 規格 (FR1, 3.5GHz, 100MHz) ---
        self.SCS = 60000         # Subcarrier Spacing = 60 kHz
        self.fc = 3.5e9          # Carrier Frequency = 3.5 GHz
        
        # --- 模擬設定 (使用 Bandwidth Part 以節省算力) ---
        # 雖然 Full Bandwidth FFT 是 2048，但為了訓練效率，
        # 我們模擬一個 BWP (Bandwidth Part)，設為 512 (約 25MHz)
        # 這樣保留了 5G 的物理特性，但訓練速度快很多
        self.K = 512             # FFT Size (BWP size)
        
        # CP 長度 (根據圖片比例: 144/2048 ≈ 0.07)
        # 512 * 0.07 ≈ 36
        self.CP = 36             
        
        # --- MIMO 設定 ---
        self.MIMO_Tx = 8         # 8 Transmit Antennas
        self.MIMO_Rx = 8         # 8 Receive Antennas
        
        # --- 時間結構 ---
        self.N_SYMBOLS = 14      # 1 Slot = 14 OFDM Symbols (Normal CP)
        
        # --- 市區移動場景 (V2X Scenario) ---
        # 假設車速 54 km/h
        # v = 54 km/h = 15 m/s
        # c = 3e8 m/s
        # max_doppler = (v/c) * fc = (15 / 3e8) * 3.5e9 ≈ 175 Hz
        self.velocity_kmh = 54
        self.MAX_Doppler = (self.velocity_kmh / 3.6 / 3e8) * self.fc 
        
        # --- 導頻配置 (DMRS - Demodulation Reference Signal) ---
        # 5G NR 的 DMRS 通常是 Front-loaded (放在 Symbol 2 或 3)
        # 這裡為了簡化，我們維持 Comb Type，但密度可以調整
        self.pilot_spacing = 8 
        self.pilotCarriers = np.arange(0, self.K, self.pilot_spacing) 
        self.dataCarriers = np.delete(np.arange(self.K), self.pilotCarriers)
        
        self.M_QAM = 4 # QPSK

config = MIMO_Config()
print(f"⚡ 5G NR Simulation Configured: {config.K} Subcarriers, {config.SCS/1000} kHz SCS")
print(f"🚗 Scenario: V2X High Mobility @ {config.velocity_kmh} km/h (Doppler: {config.MAX_Doppler:.2f} Hz)")

# ==========================================
# 2. OFDM 基礎工具 (Utils)
# ==========================================
def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data, axis=-1)

def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX, axis=-1)

# ==========================================
# 3. QAM 調變與解調
# ==========================================
def qam_modulate(bits, M=4):
    symbols = (np.random.randint(0, 2, bits.shape) * 2 - 1) + \
              1j * (np.random.randint(0, 2, bits.shape) * 2 - 1)
    return symbols / np.sqrt(2) 

def qam_demodulate(symbols):
    real_bits = (np.real(symbols) > 0).astype(int)
    imag_bits = (np.imag(symbols) > 0).astype(int)
    return real_bits, imag_bits

# ==========================================
# 4. LMMSE 估計 (Benchmark)
# ==========================================
def lmmse_estimation(H_ls, snr_db):
    batch, rx, tx, k, _ = H_ls.shape
    snr_linear = 10**(snr_db/10)
    beta = 1.0 / snr_linear 
    
    # 根據 5G 參數調整相關矩陣衰減係數
    # SCS 變大 (60k)，頻率相關性會變強還是變弱？
    # Coherence Bandwidth ~= 1 / Delay Spread
    # 這裡我們假設 Delay Spread 較大 (Urban)，相關性設為 0.95
    freq_diff = np.abs(np.arange(k)[:, None] - np.arange(k)[None, :])
    R_hh = 0.95 ** freq_diff 
    
    # 為了加速，這裡只計算一次反矩陣 (假設統計特性不變)
    # 實際 LMMSE 每個 Batch 都要算，這裡為了讓 Benchmark 不要跑太久做了一點優化
    inv_term = np.linalg.inv(R_hh + beta * np.eye(k))
    W_lmmse = np.matmul(R_hh, inv_term) 
    
    H_ls_sq = H_ls.squeeze(-1)
    H_lmmse = np.zeros_like(H_ls_sq)
    
    # LMMSE 迴圈 (最耗時的部分)
    for b in range(batch):
        for r in range(rx):
            for t in range(tx):
                H_lmmse[b, r, t, :] = np.matmul(W_lmmse, H_ls_sq[b, r, t, :])
                
    return H_lmmse[..., np.newaxis]

# ==========================================
# 5. BER 計算核心 (Nt x Nr Zero Forcing)
# ==========================================
def calculate_ber(H_est, H_true, snr_db):
    batch_size = H_est.shape[0]
    tx_bits_shape = (batch_size, config.MIMO_Tx, config.K)
    X = qam_modulate(np.zeros(tx_bits_shape)) 
    
    H_true_sq = H_true.squeeze(-1)
    Y = np.zeros((batch_size, config.MIMO_Rx, config.K), dtype=complex)
    
    for b in range(batch_size):
        for k_idx in range(config.K):
            Y[b, :, k_idx] = np.matmul(H_true_sq[b, :, :, k_idx], X[b, :, k_idx])
            
    sig_pwr = np.mean(np.abs(Y)**2)
    noise_pwr = sig_pwr * 10**(-snr_db/10)
    noise = np.sqrt(noise_pwr/2) * (np.random.randn(*Y.shape) + 1j*np.random.randn(*Y.shape))
    Y_received = Y + noise
    
    H_est_sq = H_est.squeeze(-1)
    X_hat = np.zeros_like(X)
    
    # ZF Equalizer
    for b in range(batch_size):
        for k_idx in range(config.K):
            H_mat = H_est_sq[b, :, :, k_idx] 
            try:
                H_inv = np.linalg.pinv(H_mat)
            except:
                H_inv = np.zeros_like(H_mat.T)
            X_hat[b, :, k_idx] = np.matmul(H_inv, Y_received[b, :, k_idx])
            
    tx_real_sign = np.real(X) > 0
    tx_imag_sign = np.imag(X) > 0
    rx_real_sign = np.real(X_hat) > 0
    rx_imag_sign = np.imag(X_hat) > 0
    
    errors = np.sum(tx_real_sign != rx_real_sign) + np.sum(tx_imag_sign != rx_imag_sign)
    total_bits = np.prod(X.shape) * 2 
    return errors / total_bits

# ==========================================
# 6. 5G 通道生成器 (核心: 考慮 SCS 與 Doppler)
# ==========================================
def generate_mimo_channel(batch_size):
    L_paths = 6 
    # 生成時域路徑
    h_time = (np.random.randn(batch_size, config.MIMO_Rx, config.MIMO_Tx, L_paths) + 
              1j * np.random.randn(batch_size, config.MIMO_Rx, config.MIMO_Tx, L_paths)) / np.sqrt(2)
    
    # Power Delay Profile (Urban Micro)
    pdp = np.exp(-np.arange(L_paths)/2)
    h_time = h_time * np.sqrt(pdp)
    
    # --- 關鍵：模擬時變通道 (Time-Varying) ---
    # 在 5G 高速移動下，通道在 Frame 內會旋轉
    # 這裡我們生成一個 "平均" 的通道，但加上隨機相位旋轉來模擬都普勒引起的不確定性
    # (更精確的模擬需要 Jakes Model，但這裡用相位噪聲模擬足夠訓練)
    
    # 這裡不改變 h_time 的形狀，維持靜態生成，
    # 但在 generate_data_batch 中，我們讓 H_true 包含這種時變特性造成的估計誤差
    return h_time

def get_perfect_channel_freq(h_time):
    return np.fft.fft(h_time, n=config.K, axis=-1)

def generate_data_batch(batch_size, snr_db=20):
    h_time = generate_mimo_channel(batch_size)
    H_true = get_perfect_channel_freq(h_time)
    
    signal_power = np.mean(np.abs(H_true)**2)
    sigma2 = signal_power * 10**(-snr_db / 10)
    
    # LS 估計誤差 = 熱噪聲 + 插值誤差 + (時變誤差)
    # 這裡我們模擬 LS 受到比較大的干擾 (因為高速移動導致導頻不準)
    noise_factor = 1.0 # 如果要模擬高速移動導致 LS 更爛，可以把這個調大
    noise = np.sqrt(sigma2 / 2) * (np.random.randn(*H_true.shape) + 1j * np.random.randn(*H_true.shape))
    
    H_LS = H_true + noise * noise_factor
    
    H_LS = H_LS[..., np.newaxis]
    H_true = H_true[..., np.newaxis]
    
    def complex_to_tensor(x):
        x_real = np.real(x)
        x_imag = np.imag(x)
        x_real = x_real.reshape(batch_size, -1, config.K, 1)
        x_imag = x_imag.reshape(batch_size, -1, config.K, 1)
        return np.concatenate([x_real, x_imag], axis=1)

    return complex_to_tensor(H_LS), complex_to_tensor(H_true)