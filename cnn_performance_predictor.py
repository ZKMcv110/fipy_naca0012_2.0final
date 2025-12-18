#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于CNN+MLP的CFD性能参数预测模型 (最终完整版：特征融合 + 双向归一化)

解决痛点：
1. 解决 f 值过小导致的梯度消失 -> 输出标签归一化
2. 解决 预测值呈水平直线(模型躺平) -> 引入物理参数(Tt, Ts...)辅助预测
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 导入模型定义
from cnn_model_definition import CFDFieldToPerformanceCNN

# ==========================================
# 0. 环境配置
# ==========================================
# 强制指定GPU算力 (适配 RTX 50 系列)
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
os.environ["CUDA_DISABLE_DEVICE_CHECK"] = "1"

# ==========================================
# 3. 数据集类 (含输入/输出归一化)
# ==========================================
class CFDFieldDataset(Dataset):
    def __init__(self, data_dir, csv_file, transform=None, field_types=None):
        self.data_dir = data_dir
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.field_types = field_types or ['velocity_field', 'pressure_field', 'temperature_field']
        
        # 1. 验证有效目录
        self.solution_dirs = self._get_solution_dirs()
        self.valid_indices = self._validate_directories()
        print(f"有效数据样本数: {len(self.valid_indices)}")
        
        # 提取有效数据的 DataFrame
        valid_df = self.data_frame.iloc[self.valid_indices]
        
        # 2. 计算输出标签 (Target) 的统计量 - 用于归一化
        self.target_stats = {
            'nu_mean': valid_df['Nu'].mean(),
            'nu_std': valid_df['Nu'].std(),
            'f_mean': valid_df['f'].mean(),
            'f_std': valid_df['f'].std()
        }
        
        # 3. 计算输入参数 (Input Scalars) 的统计量 - 用于归一化
        # 假设物理参数列名为 Tt, Ts, Tad, Tb
        self.param_cols = ['Tt', 'Ts', 'Tad', 'Tb']
        self.param_stats = {}
        for col in self.param_cols:
            self.param_stats[f'{col}_mean'] = valid_df[col].mean()
            self.param_stats[f'{col}_std'] = valid_df[col].std()
            
        print("-" * 40)
        print("数据集统计信息 (用于归一化):")
        print(f"Target Nu: Mean={self.target_stats['nu_mean']:.2f}, Std={self.target_stats['nu_std']:.2f}")
        print(f"Target f : Mean={self.target_stats['f_mean']:.6f}, Std={self.target_stats['f_std']:.6f}")
        print("-" * 40)

    def _get_solution_dirs(self):
        all_dirs = os.listdir(self.data_dir)
        return sorted([d for d in all_dirs if d.endswith('_cfd_solution') and os.path.isdir(os.path.join(self.data_dir, d))])
        
    def _validate_directories(self):
        valid_indices = []
        field_map = {'velocity_field': 'velocity_magnitude.png', 'pressure_field': 'pressure.png', 'temperature_field': 'temperature.png'}
        
        for i in range(min(len(self.solution_dirs), len(self.data_frame))):
            sol_dir = self.solution_dirs[i]
            img_dir = os.path.join(self.data_dir, sol_dir)
            if not os.path.exists(img_dir): continue
                
            if all(os.path.exists(os.path.join(img_dir, field_map.get(ft, f'{ft}.png'))) for ft in self.field_types):
                valid_indices.append(i)
        return valid_indices
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        idx = idx % len(self)
        actual_idx = self.valid_indices[idx]
        row = self.data_frame.iloc[actual_idx]
        
        # --- A. 加载图像 ---
        sol_dir = self.solution_dirs[actual_idx]
        img_dir = os.path.join(self.data_dir, sol_dir)
        field_map = {'velocity_field': 'velocity_magnitude.png', 'pressure_field': 'pressure.png', 'temperature_field': 'temperature.png'}
        
        imgs = []
        for ft in self.field_types:
            p = os.path.join(img_dir, field_map.get(ft, f'{ft}.png'))
            if os.path.exists(p):
                img = Image.open(p).convert('RGB')
                if self.transform: img = self.transform(img)
                imgs.append(img)
            else:
                imgs.append(torch.zeros(3, 224, 224))
        images = torch.stack(imgs, dim=0)
        
        # --- B. 加载并归一化 物理参数 (Inputs) ---
        params = []
        for col in self.param_cols:
            val = float(row[col])
            mean = self.param_stats[f'{col}_mean']
            std = self.param_stats[f'{col}_std']
            # Z-Score Normalization
            norm_val = (val - mean) / (std + 1e-8)
            params.append(norm_val)
        params_tensor = torch.tensor(params, dtype=torch.float32)
        
        # --- C. 加载并归一化 性能指标 (Targets) ---
        nu_val = float(row['Nu'])
        f_val = float(row['f'])
        
        norm_nu = (nu_val - self.target_stats['nu_mean']) / (self.target_stats['nu_std'] + 1e-8)
        norm_f = (f_val - self.target_stats['f_mean']) / (self.target_stats['f_std'] + 1e-8)
        
        performance = torch.tensor([norm_nu, norm_f], dtype=torch.float32)
        
        # Case ID
        case_id = row.get('case_id', actual_idx + 1)
        
        return {
            'images': images,
            'parameters': params_tensor,
            'performance': performance,
            'case_id': case_id
        }

def collate_fn(batch):
    images = torch.stack([item['images'] for item in batch], dim=0)
    params = torch.stack([item['parameters'] for item in batch], dim=0)
    perfs = torch.stack([item['performance'] for item in batch], dim=0)
    ids = [item['case_id'] for item in batch]
    return {'images': images, 'parameters': params, 'performance': perfs, 'case_id': ids}

# ==========================================
# 4. 训练与评估流程
# ==========================================
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, device='cpu', model_dir='.'):
    train_hist, val_hist = [], []
    model.to(device)
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        run_loss = 0.0
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        for i, data in enumerate(train_loader):
            imgs = data['images'].to(device)
            params = data['parameters'].to(device) # 新增参数输入
            targets = data['performance'].to(device)
            
            optimizer.zero_grad()
            # 前向传播：传入图像 + 参数
            outputs = model(imgs, params)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * imgs.size(0)
            
        avg_train_loss = run_loss / len(train_loader.dataset)
        train_hist.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                imgs = data['images'].to(device)
                params = data['parameters'].to(device)
                targets = data['performance'].to(device)
                
                outputs = model(imgs, params)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * imgs.size(0)
        
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_hist.append(avg_val_loss)
        
        print(f"  Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            if model_dir:
                torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pth'))

    # 加载最佳模型
    if os.path.exists(os.path.join(model_dir, 'best_model.pth')):
        model.load_state_dict(torch.load(os.path.join(model_dir, 'best_model.pth')))
    return train_hist, val_hist

def evaluate_model(model, test_loader, dataset, device='cpu'):
    """评估模型并自动反归一化"""
    model.to(device)
    model.eval()
    
    true_list, pred_list = [], []
    
    with torch.no_grad():
        for data in test_loader:
            imgs = data['images'].to(device)
            params = data['parameters'].to(device)
            targets = data['performance'].to(device)
            
            outputs = model(imgs, params)
            
            true_list.extend(targets.cpu().numpy())
            pred_list.extend(outputs.cpu().numpy())
            
    true_norm = np.array(true_list)
    pred_norm = np.array(pred_list)
    
    # --- 反归一化还原真实物理值 ---
    stats = dataset.target_stats
    true_real = np.zeros_like(true_norm)
    pred_real = np.zeros_like(pred_norm)
    
    # Nu (Index 0)
    true_real[:, 0] = true_norm[:, 0] * (stats['nu_std'] + 1e-8) + stats['nu_mean']
    pred_real[:, 0] = pred_norm[:, 0] * (stats['nu_std'] + 1e-8) + stats['nu_mean']
    
    # f (Index 1)
    true_real[:, 1] = true_norm[:, 1] * (stats['f_std'] + 1e-8) + stats['f_mean']
    pred_real[:, 1] = pred_norm[:, 1] * (stats['f_std'] + 1e-8) + stats['f_mean']
    
    # 计算误差
    metrics = {
        'Nu_MSE': mean_squared_error(true_real[:, 0], pred_real[:, 0]),
        'f_MSE': mean_squared_error(true_real[:, 1], pred_real[:, 1]),
        'Nu_MAE': mean_absolute_error(true_real[:, 0], pred_real[:, 0]),
        'f_MAE': mean_absolute_error(true_real[:, 1], pred_real[:, 1])
    }
    
    return metrics, true_real, pred_real

def plot_results(true_vals, pred_vals, save_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot Nu
    ax1.scatter(true_vals[:, 0], pred_vals[:, 0], alpha=0.6, c='blue')
    min_v, max_v = true_vals[:, 0].min(), true_vals[:, 0].max()
    ax1.plot([min_v, max_v], [min_v, max_v], 'r--', lw=2)
    ax1.set_title(f'Nu Prediction (MAE: {mean_absolute_error(true_vals[:,0], pred_vals[:,0]):.2f})')
    ax1.set_xlabel('True Nu'); ax1.set_ylabel('Predicted Nu')
    ax1.grid(True)
    
    # Plot f
    ax2.scatter(true_vals[:, 1], pred_vals[:, 1], alpha=0.6, c='green')
    min_v, max_v = true_vals[:, 1].min(), true_vals[:, 1].max()
    ax2.plot([min_v, max_v], [min_v, max_v], 'r--', lw=2)
    ax2.set_title(f'f Prediction (MAE: {mean_absolute_error(true_vals[:,1], pred_vals[:,1]):.6f})')
    ax2.set_xlabel('True f'); ax2.set_ylabel('Predicted f')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'final_prediction.png'))
    plt.close()

# ==========================================
# 5. 主函数
# ==========================================
def main():
    print(">>> 启动 CFD 参数预测训练 (Hybrid CNN+MLP) <<<")
    # 使用CUDA如果可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"计算设备: {device}")
    
    # 路径设置
    results_dir = 'results'
    csv_file = os.path.join('csv_data', 'final_results.csv')
    model_dir = 'ai_cnn_model_results'
    os.makedirs(model_dir, exist_ok=True)
    
    if not os.path.exists(csv_file):
        print(f"Error: 找不到 {csv_file}")
        return

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print("\n1. 初始化数据集...")
    # 假设输入有4个参数: Tt, Ts, Tad, Tb
    dataset = CFDFieldDataset(results_dir, csv_file, transform, field_types=['velocity_field', 'pressure_field', 'temperature_field'])
    
    if len(dataset) == 0: return
    
    # 划分数据集
    train_size = int(0.75 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    
    loaders = {
        'train': DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=0),
        'val': DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=0),
        'test': DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=0)
    }
    
    print("\n2. 初始化混合模型...")
    # num_fields=3 (图像), num_scalars=4 (Tt,Ts,Tad,Tb)
    model = CFDFieldToPerformanceCNN(num_fields=3, num_scalars=4, output_size=2)
    
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    print("\n3. 开始训练...")
    train_hist, val_hist = train_model(model, loaders['train'], loaders['val'], criterion, optimizer, num_epochs=10, device=device, model_dir=model_dir)
    
    # 绘制Loss曲线
    plt.plot(train_hist, label='Train'); plt.plot(val_hist, label='Val')
    plt.legend(); plt.savefig(os.path.join(model_dir, 'loss_curve.png')); plt.close()
    
    print("\n4. 最终评估...")
    metrics, true_vals, pred_vals = evaluate_model(model, loaders['test'], dataset, device)
    
    print("-" * 30)
    # 将NumPy数值转换为Python原生类型以便JSON序列化
    serializable_metrics = {k: float(v) for k, v in metrics.items()}
    for k, v in serializable_metrics.items(): print(f"{k}: {v:.6f}")
    print("-" * 30)
    
    # 保存结果
    with open(os.path.join(model_dir, 'metrics.json'), 'w') as f: json.dump(serializable_metrics, f, indent=2)
    plot_results(true_vals, pred_vals, model_dir)
    torch.save(model.state_dict(), os.path.join(model_dir, 'cfd_cnn_model.pth'))
    
    print(f"\n全部完成！查看目录: {model_dir}")

if __name__ == '__main__':
    main()