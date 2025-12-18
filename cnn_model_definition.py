#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CNN模型定义文件，供训练和预测脚本共同使用
"""

import torch
import torch.nn as nn
import os

# ==========================================
# 1. 注意力机制模块 (CBAM)
# ==========================================
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

# ==========================================
# 2. CFD性能预测CNN模型
# ==========================================
class CFDFieldToPerformanceCNN(nn.Module):
    def __init__(self, num_fields=3, num_scalars=4, output_size=2):
        """
        参数:
        num_fields: 输入流场图数量 (默认3: 速度, 压力, 温度)
        num_scalars: 输入物理标量数量 (默认4: Tt, Ts, Tad, Tb)
        output_size: 输出预测值数量 (默认2: Nu, f)
        """
        super(CFDFieldToPerformanceCNN, self).__init__()
        
        # --- A. 图像特征提取 (CNN) ---
        self.cnn_extractors = nn.ModuleList([
            nn.Sequential(
                # Block 1
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
                # Block 2
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
                # Block 3
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
                # Attention
                CBAM(128),
                # Block 4
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256), nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)) # 输出大小: 256 x 4 x 4
            ) for _ in range(num_fields)
        ])
        
        # 计算图像展平后的特征维度
        self.img_flat_dim = 256 * num_fields * 4 * 4  # 12288
        
        # --- B. 物理参数特征提取 (MLP) ---
        self.scalar_mlp = nn.Sequential(
            nn.Linear(num_scalars, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        # --- C. 特征融合与预测 ---
        # 输入 = 图像特征(12288) + 物理参数特征(128)
        combined_dim = self.img_flat_dim + 128
        
        self.predictor = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_size) # 输出 Nu 和 f
        )
        
    def forward(self, images, scalars):
        batch_size = images.size(0)
        num_fields = images.size(1)
        
        # 1. 处理图像
        img_features_list = []
        for i in range(num_fields):
            # 提取第 i 个物理场的特征
            feat = self.cnn_extractors[i](images[:, i, :, :, :])
            # 展平: (Batch, 256*4*4)
            feat = feat.view(batch_size, -1)
            img_features_list.append(feat)
        
        # 拼接所有物理场的特征
        combined_img_feat = torch.cat(img_features_list, dim=1)
        
        # 2. 处理标量参数
        scalar_feat = self.scalar_mlp(scalars)
        
        # 3. 特征融合 (拼接)
        total_feat = torch.cat([combined_img_feat, scalar_feat], dim=1)
        
        # 4. 最终预测
        output = self.predictor(total_feat)
        return output