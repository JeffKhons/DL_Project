"""
Project: Deep Learning for MIMO-OFDM Channel Estimation
File: dccrn_module.py
Author: Jeff K Chen
Date: 2025

Description:
    This module implements a Deep Complex Convolutional Network (DCCRN), an architecture 
    originally renowned in audio speech enhancement, now adapted for wireless mMIMO-OFDM 
    channel estimation.

    Motivation & Design:
    - Cross-Domain Adaptation: By treating the Channel Frequency Response (CFR) similarly 
      to complex audio spectrograms, this model leverages the DCCRN architecture to 
      capture frequency correlations in wireless channels.
    - Complex-Valued Learning: Unlike standard real-valued networks, this model explicitly 
      operates on complex numbers using specialized layers. This is crucial for effectively 
      learning complex spectral features and preserving phase information, which are 
      intrinsic to accurate channel estimation.
"""

import torch
import torch.nn as nn

# ==========================================
# 1. 複數運算組件
# ==========================================
# ... (rest of the code)

import torch
import torch.nn as nn

# ==========================================
# 1. 複數運算組件
# ==========================================
class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ComplexConv2d, self).__init__()
        self.conv_re = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_im = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        ch = x.shape[1] // 2
        real, imag = x[:, :ch], x[:, ch:]
        out_real = self.conv_re(real) - self.conv_im(imag)
        out_imag = self.conv_re(imag) + self.conv_im(real)
        return torch.cat([out_real, out_imag], dim=1)

class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super(ComplexBatchNorm2d, self).__init__()
        self.bn_re = nn.BatchNorm2d(num_features)
        self.bn_im = nn.BatchNorm2d(num_features)

    def forward(self, x):
        ch = x.shape[1] // 2
        real, imag = x[:, :ch], x[:, ch:]
        real = self.bn_re(real)
        imag = self.bn_im(imag)
        return torch.cat([real, imag], dim=1)

# ==========================================
# 2. DCCRN 主模型 (天線 8x8 & 4 Layers)
# ==========================================
class DCCRN_Net(nn.Module):
    def __init__(self, mimo_rx=8, mimo_tx=8):
        super(DCCRN_Net, self).__init__()
        self.in_ch = mimo_rx * mimo_tx 
        
        # --- Encoder (4層) ---
        # [MODIFIED] Layer 1: 維持 64 通道
        self.enc1 = nn.Sequential(
            ComplexConv2d(self.in_ch, 64, kernel_size=(5,1), padding=(2,0)), 
            ComplexBatchNorm2d(64),
            nn.PReLU()
        )
        # [MODIFIED] Layer 2: 64 -> 128
        self.enc2 = nn.Sequential(
            ComplexConv2d(64, 128, kernel_size=(3,1), stride=(2,1), padding=(1,0)),
            ComplexBatchNorm2d(128),
            nn.PReLU()
        )
        # [MODIFIED] Layer 3: 128 -> 256
        self.enc3 = nn.Sequential(
            ComplexConv2d(128, 256, kernel_size=(3,1), stride=(2,1), padding=(1,0)),
            ComplexBatchNorm2d(256),
            nn.PReLU()
        )
        # [NEW] Layer 4: 新增一層深層特徵提取 (256 -> 256)
        # 這層能幫助模型在極低 SNR 下提取抽象特徵
        self.enc4 = nn.Sequential(
            ComplexConv2d(256, 256, kernel_size=(3,1), stride=(2,1), padding=(1,0)),
            ComplexBatchNorm2d(256),
            nn.PReLU()
        )
        
        # --- Decoder (對稱加寬 + 跳躍連接) ---
        # [NEW] Decoder 4 (對應 Encoder 4)
        self.dec4 = nn.Sequential(
            nn.Upsample(scale_factor=(2,1), mode='bilinear', align_corners=False),
            ComplexConv2d(256 + 256, 256, kernel_size=(3,1), padding=(1,0)), # Skip: 256+256 -> 256
            ComplexBatchNorm2d(256),
            nn.PReLU()
        )
        # [MODIFIED] Decoder 3
        self.dec3 = nn.Sequential(
            nn.Upsample(scale_factor=(2,1), mode='bilinear', align_corners=False),
            ComplexConv2d(256 + 256, 128, kernel_size=(3,1), padding=(1,0)),
            ComplexBatchNorm2d(128),
            nn.PReLU()
        )
        # [MODIFIED] Decoder 2
        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=(2,1), mode='bilinear', align_corners=False),
            ComplexConv2d(128 + 128, 64, kernel_size=(3,1), padding=(1,0)),
            ComplexBatchNorm2d(64),
            nn.PReLU()
        )
        # [MODIFIED] Decoder 1 (Output)
        self.dec1 = nn.Sequential(
            ComplexConv2d(64 + 64, 64, kernel_size=(3,1), padding=(1,0)),
            ComplexBatchNorm2d(64),
            nn.PReLU(),
            ComplexConv2d(64, self.in_ch, kernel_size=(3,1), padding=(1,0))
        )
        
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3) # [NEW] Forward pass for new layer
        
        # Skip Connection with new layer
        d4 = self.dec4(torch.cat([e4, e4], dim=1)) # [NEW] Center bottleneck
        d3 = self.dec3(torch.cat([d4, e3], dim=1)) 
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        
        return d1