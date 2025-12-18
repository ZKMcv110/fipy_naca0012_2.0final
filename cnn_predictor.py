#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用训练好的CNN模型预测CFD性能参数
"""

import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# 导入CNN模型定义
from cnn_model_definition import CFDFieldToPerformanceCNN

def load_trained_model(model_path, device):
    """加载训练好的CNN模型"""
    # 创建模型实例
    model = CFDFieldToPerformanceCNN(num_fields=3, output_size=2)
    
    # 加载模型权重
    if os.path.exists(model_path):
        try:
            # 首先尝试直接加载（适用于新模型结构）
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            print(f"成功加载模型: {model_path}")
            return model
        except RuntimeError as e:
            # 如果直接加载失败，尝试兼容旧模型结构
            print("尝试兼容旧模型结构...")
            try:
                # 加载权重字典
                state_dict = torch.load(model_path, map_location=device)
                
                # 获取当前模型的状态字典
                model_dict = model.state_dict()
                
                # 过滤掉不匹配的键
                filtered_dict = {}
                for k, v in state_dict.items():
                    if k in model_dict and v.shape == model_dict[k].shape:
                        filtered_dict[k] = v
                    else:
                        print(f"跳过不匹配的键: {k}")
                
                # 加载过滤后的权重
                model_dict.update(filtered_dict)
                model.load_state_dict(model_dict, strict=False)
                model.to(device)
                model.eval()
                print(f"成功以兼容模式加载模型: {model_path}")
                print(f"加载了 {len(filtered_dict)} 个权重参数")
                return model
            except Exception as e2:
                print(f"兼容模式加载失败: {e2}")
                return None
    else:
        print(f"错误: 找不到模型文件 {model_path}")
        return None

def preprocess_field_images(velocity_img_path, pressure_img_path, temperature_img_path, device):
    """预处理场变量图像"""
    # 定义图像变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet标准化
    ])
    
    # 加载和预处理图像
    try:
        vel_img = Image.open(velocity_img_path).convert('RGB')
        pres_img = Image.open(pressure_img_path).convert('RGB')
        temp_img = Image.open(temperature_img_path).convert('RGB')
        
        vel_tensor = transform(vel_img)
        pres_tensor = transform(pres_img)
        temp_tensor = transform(temp_img)
        
        # 堆叠图像
        images = torch.stack([vel_tensor, pres_tensor, temp_tensor], dim=0)
        images = images.unsqueeze(0)  # 添加batch维度
        images = images.to(device)
        
        return images
    except Exception as e:
        print(f"图像预处理失败: {e}")
        return None

def predict_performance(model, images):
    """使用CNN模型预测性能参数"""
    with torch.no_grad():
        # 对于预测，我们不需要物理参数，所以创建一个零张量作为占位符
        dummy_params = torch.zeros(1, 4).to(images.device)
        outputs = model(images, dummy_params)
        nu_pred, f_pred = outputs[0]
        return nu_pred.item(), f_pred.item()

def visualize_field_images(velocity_img_path, pressure_img_path, temperature_img_path, save_path=None):
    """可视化场变量图像"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 显示速度场
    vel_img = Image.open(velocity_img_path)
    axes[0].imshow(vel_img)
    axes[0].set_title('Velocity Field')
    axes[0].axis('off')
    
    # 显示压力场
    pres_img = Image.open(pressure_img_path)
    axes[1].imshow(pres_img)
    axes[1].set_title('Pressure Field')
    axes[1].axis('off')
    
    # 显示温度场
    temp_img = Image.open(temperature_img_path)
    axes[2].imshow(temp_img)
    axes[2].set_title('Temperature Field')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """主函数"""
    print("使用CNN模型预测CFD性能参数...")
    
    # 检查CUDA可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 模型路径
    model_dir = 'ai_cnn_model_results'
    model_path = os.path.join(model_dir, 'cfd_cnn_model.pth')
    
    # 检查模型是否存在
    if not os.path.exists(model_path):
        print(f"错误: 找不到训练好的CNN模型 {model_path}")
        print("请先运行 cnn_performance_predictor.py 训练模型")
        return
    
    # 加载模型
    model = load_trained_model(model_path, device)
    if model is None:
        return
    
    # 示例：使用results目录中的最新案例进行预测
    # 在实际应用中，这些应该是真实的CFD场变量图像路径
    print("\n查找最新的CFD仿真结果...")
    
    # 查找results目录中的最新案例
    results_dir = 'results'
    if not os.path.exists(results_dir):
        print(f"错误: 找不到结果目录 {results_dir}")
        return
    
    # 查找所有_cfd_solution目录
    solution_dirs = []
    for root, dirs, files in os.walk(results_dir):
        for d in dirs:
            if d.endswith('_cfd_solution'):
                solution_dirs.append(os.path.join(root, d))
    
    # 统计找到的物理场图像数据组数
    total_image_groups = len(solution_dirs)
    print(f"找到 {total_image_groups} 组物理场图像数据")
    
    # 检查是否有可用的案例
    if not solution_dirs:
        print("错误: 没有找到任何CFD解决方案目录")
        return
    
    # 选择最新的解决方案目录
    latest_solution_dir = max(solution_dirs, key=os.path.getmtime)
    print(f"使用解决方案目录: {latest_solution_dir}")
    
    # 明确说明案例的来源
    print(f"该案例来自CFD仿真结果，对应于results目录下的最新仿真案例")
    print(f"物理场图像文件位于: {latest_solution_dir}")
    print(f"速度场: {os.path.join(latest_solution_dir, 'velocity_magnitude.png')}")
    print(f"压力场: {os.path.join(latest_solution_dir, 'pressure.png')}")
    print(f"温度场: {os.path.join(latest_solution_dir, 'temperature.png')}")
    
    # 图像路径 (使用实际生成的文件名)
    velocity_img_path = os.path.join(latest_solution_dir, 'velocity_magnitude.png')
    pressure_img_path = os.path.join(latest_solution_dir, 'pressure.png')
    temperature_img_path = os.path.join(latest_solution_dir, 'temperature.png')
    
    # 检查图像文件是否存在
    missing_files = []
    for img_path in [velocity_img_path, pressure_img_path, temperature_img_path]:
        if not os.path.exists(img_path):
            missing_files.append(img_path)
    
    if missing_files:
        print("错误: 缺少以下图像文件:")
        for file in missing_files:
            print(f"  - {file}")
        return
    
    # 预处理图像
    print("预处理场变量图像...")
    images = preprocess_field_images(
        velocity_img_path, pressure_img_path, temperature_img_path, device
    )
    
    if images is None:
        print("图像预处理失败")
        return
    
    # 进行预测
    print("使用CNN模型进行预测...")
    nu_pred, f_pred = predict_performance(model, images)
    
    print(f"\n预测结果:")
    print(f"  Nu (努塞尔数): {nu_pred:.4f}")
    print(f"  f  (摩擦因子): {f_pred:.4f}")
    if f_pred > 0:
        target_param = nu_pred / (f_pred**(1/3))
        print(f"  Nu/(f^(1/3)): {target_param:.4f}")
    else:
        print("  无法计算目标参数 (f <= 0)")
    
    # 保存预测结果
    prediction_result = {
        'Nu_predicted': nu_pred,
        'f_predicted': f_pred,
        'target_parameter': nu_pred / (f_pred**(1/3)) if f_pred > 0 else 0
    }
    
    import json
    prediction_file = os.path.join(model_dir, 'prediction_result.json')
    with open(prediction_file, 'w') as f:
        json.dump(prediction_result, f, indent=2)
    print(f"预测结果已保存到: {prediction_file}")
    
    # 可视化场变量图像
    visualization_path = os.path.join(model_dir, 'predicted_field_visualization.png')
    visualize_field_images(
        velocity_img_path, pressure_img_path, temperature_img_path, 
        visualization_path
    )
    print(f"场变量可视化已保存到: {visualization_path}")
    
    # 明确说明所有预测结果的存储位置
    print(f"\n所有预测结果已保存到目录: {model_dir}")
    print(f"  - 预测数据: {prediction_file}")
    print(f"  - 场域可视化图: {visualization_path}")

if __name__ == "__main__":
    main()