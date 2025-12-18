#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPU和cuDNN状态检查与性能测试脚本
用于验证深度学习环境是否正确配置并充分利用硬件加速
"""

import torch
import torch.nn as nn
import time

def check_basic_gpu_status():
    """检查基础GPU状态"""
    print("=" * 60)
    print("GPU和CUDA基础信息检查")
    print("=" * 60)
    
    # 基础：PyTorch版本
    print("PyTorch版本：", torch.__version__)

    # CUDA核心信息
    print("CUDA是否可用：", torch.cuda.is_available())
    print("PyTorch绑定的CUDA版本：", torch.version.cuda if torch.cuda.is_available() else "无CUDA")

    # cuDNN版本（关键补充）
    if torch.cuda.is_available():
        print("cuDNN版本：", torch.backends.cudnn.version())
        print("cuDNN是否启用：", torch.backends.cudnn.enabled)
    else:
        print("cuDNN版本：无（CPU版PyTorch不包含cuDNN）")

    # 显卡硬件信息
    if torch.cuda.is_available():
        print("显卡名称：", torch.cuda.get_device_name(0))
        print("显卡数量：", torch.cuda.device_count())
        print("当前使用显卡ID：", torch.cuda.current_device())
        print("显卡算力：", torch.cuda.get_device_capability(0))  # 可选：查看显卡算力
    else:
        print("显卡名称：无GPU")
    
    print()

def test_cudnn_performance():
    """测试cuDNN性能加速效果"""
    if not torch.cuda.is_available():
        print("CUDA不可用，无法进行cuDNN性能测试")
        return
    
    print("=" * 60)
    print("cuDNN性能加速测试")
    print("=" * 60)
    
    device = torch.device('cuda')
    print(f"设备: {torch.cuda.get_device_name()}")
    
    # 创建测试数据
    batch_size = 32
    channels = 64
    height, width = 224, 224
    x = torch.randn(batch_size, channels, height, width, device=device)
    
    # 创建卷积层
    conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1).to(device)
    
    # 测试多次运行取平均值
    num_iterations = 50
    
    # 1. 启用cuDNN测试
    torch.backends.cudnn.enabled = True
    print(f"cuDNN状态: {'启用' if torch.backends.cudnn.enabled else '禁用'}")
    print(f"cuDNN版本: {torch.backends.cudnn.version()}")
    
    # 预热
    for _ in range(10):
        _ = conv(x)
    torch.cuda.synchronize()
    
    # 实际测试
    start_time = time.time()
    for _ in range(num_iterations):
        y = conv(x)
    torch.cuda.synchronize()
    time_with_cudnn = time.time() - start_time
    print(f"启用cuDNN运行时间: {time_with_cudnn:.4f}秒")
    
    # 2. 禁用cuDNN测试
    torch.backends.cudnn.enabled = False
    print(f"cuDNN状态: {'启用' if torch.backends.cudnn.enabled else '禁用'}")
    
    # 预热
    for _ in range(10):
        _ = conv(x)
    torch.cuda.synchronize()
    
    # 实际测试
    start_time = time.time()
    for _ in range(num_iterations):
        y = conv(x)
    torch.cuda.synchronize()
    time_without_cudnn = time.time() - start_time
    print(f"禁用cuDNN运行时间: {time_without_cudnn:.4f}秒")
    
    # 计算性能差异
    speedup = time_without_cudnn / time_with_cudnn
    print(f"\n性能对比:")
    print(f"  启用cuDNN: {time_with_cudnn:.4f}秒")
    print(f"  禁用cuDNN: {time_without_cudnn:.4f}秒")
    print(f"  加速比: {speedup:.2f}x")
    
    if speedup > 1.1:
        print("✓ cuDNN提供了显著的性能加速")
    elif speedup > 1.05:
        print("✓ cuDNN提供了轻微的性能加速")
    else:
        print("⚠ cuDNN未提供明显性能提升（可能原因：操作太简单或数据量太小）")
    
    # 恢复cuDNN设置
    torch.backends.cudnn.enabled = True
    print()

def test_batch_norm_cudnn():
    """测试批归一化层的cuDNN加速"""
    if not torch.cuda.is_available():
        print("CUDA不可用，无法进行批归一化cuDNN测试")
        return
    
    print("=" * 60)
    print("批归一化层cuDNN测试")
    print("=" * 60)
    
    device = torch.device('cuda')
    print(f"设备: {torch.cuda.get_device_name()}")
    
    # 创建测试数据
    batch_size = 64
    channels = 128
    height, width = 56, 56
    x = torch.randn(batch_size, channels, height, width, device=device)
    
    # 创建批归一化层
    bn = nn.BatchNorm2d(channels).to(device)
    
    # 测试多次运行取平均值
    num_iterations = 50
    
    # 1. 启用cuDNN测试
    torch.backends.cudnn.enabled = True
    
    # 预热
    for _ in range(10):
        _ = bn(x)
    torch.cuda.synchronize()
    
    # 实际测试
    start_time = time.time()
    for _ in range(num_iterations):
        y = bn(x)
    torch.cuda.synchronize()
    time_with_cudnn = time.time() - start_time
    print(f"启用cuDNN运行时间: {time_with_cudnn:.4f}秒")
    
    # 2. 禁用cuDNN测试
    torch.backends.cudnn.enabled = False
    
    # 预热
    for _ in range(10):
        _ = bn(x)
    torch.cuda.synchronize()
    
    # 实际测试
    start_time = time.time()
    for _ in range(num_iterations):
        y = bn(x)
    torch.cuda.synchronize()
    time_without_cudnn = time.time() - start_time
    print(f"禁用cuDNN运行时间: {time_without_cudnn:.4f}秒")
    
    # 计算性能差异
    speedup = time_without_cudnn / time_with_cudnn
    print(f"\n性能对比:")
    print(f"  启用cuDNN: {time_with_cudnn:.4f}秒")
    print(f"  禁用cuDNN: {time_without_cudnn:.4f}秒")
    print(f"  加速比: {speedup:.2f}x")
    
    if speedup > 1.1:
        print("✓ cuDNN为批归一化提供了显著的性能加速")
    elif speedup > 1.05:
        print("✓ cuDNN为批归一化提供了轻微的性能加速")
    else:
        print("⚠ cuDNN未为批归一化提供明显性能提升")
    
    # 恢复cuDNN设置
    torch.backends.cudnn.enabled = True
    print()

def main():
    """主函数"""
    print("GPU和cuDNN状态检查与性能测试")
    
    # 检查基础GPU状态
    check_basic_gpu_status()
    
    # 测试cuDNN性能
    test_cudnn_performance()
    
    # 测试批归一化cuDNN加速
    test_batch_norm_cudnn()
    
    print("=" * 60)
    print("所有测试完成")
    print("=" * 60)

if __name__ == "__main__":
    main()