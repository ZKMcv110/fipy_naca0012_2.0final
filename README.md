# NACA0012翼型阵列CFD仿真与优化项目

本项目基于FiPy实现了NACA0012翼型阵列的CFD仿真和参数优化。项目包括几何生成、网格划分、CFD求解、结果后处理和AI优化等模块。

## 项目结构

```
fipy_naca0012_2.0/
├── naca0012dat.py              # 翼型几何生成脚本
├── 网格生成器2_三角形.py        # 网格生成脚本
├── 求解器_pvtnf.py             # CFD求解器
├── generate_samples.py         # 参数采样生成脚本
├── run_all_cases.py            # 批量运行所有案例
├── train_ai_models.py          # 训练AI代理模型
├── ai_optimization.py          # 使用AI进行参数优化
├── cnn_performance_predictor.py # 基于CNN的性能预测模型
├── predict_performance_with_cnn.py # 使用CNN进行性能预测
├── csv_data/                   # CSV数据文件目录
├── results/                    # 仿真结果目录
├── solver_results/             # 求解器结果目录
├── ai_model_results/           # AI模型结果目录
└── ai_cnn_model_results/       # CNN模型结果目录
```

## 功能模块

### 1. 几何生成
使用 `naca0012dat.py` 生成NACA0012翼型阵列的几何数据文件。

### 2. 网格生成
使用 `网格生成器2_三角形.py` 基于几何数据生成CFD计算网格。

### 3. CFD求解
使用 `求解器_pvtnf.py` 求解Navier-Stokes方程和能量方程。

### 4. 参数采样
使用 `generate_samples.py` 生成参数组合用于批量仿真。

### 5. 批量运行
使用 `run_all_cases.py` 批量运行所有参数组合的仿真。

### 6. AI代理模型
使用 `train_ai_models.py` 和 `ai_optimization.py` 训练和使用AI模型进行优化。

### 7. CNN性能预测
使用 `cnn_performance_predictor.py` 和 `predict_performance_with_cnn.py` 实现基于CNN的性能参数预测。

## 使用流程

### 基本CFD仿真流程
1. 生成参数样本: `python generate_samples.py`
2. 批量运行案例: `python run_all_cases.py`
3. 训练AI模型: `python train_ai_models.py`
4. 进行优化: `python ai_optimization.py`

### CNN性能预测流程
1. 运行CFD仿真生成场变量图像（求解器会自动输出图像）
   - 速度场图像: `velocity_field.png`
   - 压力场图像: `pressure_field.png`
   - 温度场图像: `temperature_field.png`
2. 训练CNN模型: `python cnn_performance_predictor.py`
3. 使用CNN进行预测: `python predict_performance_with_cnn.py`

## 参数说明

项目中涉及6个几何参数：
- Tt: 横向间距系数
- Ts: 纵向间距系数
- Ta: 弯度系数（固定为0）
- Tad: 交错间距系数
- Twa: 宽长比（不参与计算）
- Tb: 厚度系数

## 输出文件

- `csv_data/final_results.csv`: 包含所有案例的参数和性能结果
- `results/`: 包含详细的CFD仿真结果，包括场变量图像
- `ai_model_results/`: 包含AI代理模型和相关文件
- `ai_cnn_model_results/`: 包含CNN模型和相关文件

## 依赖库

- FiPy: CFD求解器
- Gmsh: 网格生成
- PyTorch: AI模型训练
- Scikit-learn: 数据预处理
- NumPy, Pandas: 数据处理
- Matplotlib: 结果可视化

## 许可证

本项目采用MIT许可证。