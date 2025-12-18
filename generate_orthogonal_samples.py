#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
生成正交试验设计样本点用于优化

该脚本将使用正交表方法生成参数组合，适用于因素筛选和主效应分析
"""

import numpy as np
import pandas as pd
import os

def generate_simple_orthogonal():
    """
    生成简单的正交试验设计样本点
    
    这个版本使用了最简单的正交表L9(3^4)
    """
    
    # 定义参数范围
    bounds = {
        'Tt': [0.5, 1.5],  # 横向间距系数
        'Ts': [0.5, 1.5],  # 纵向间距系数
        'Tad': [0.0, 1.0], # 交错间距系数
        'Twa': [0.1, 1.0], # 宽长比
        'Tb': [0.06, 0.24] # 厚度系数
    }
    
    # 创建一个简单的正交表 L9(3^4)
    # 9次实验，4个因素，每个因素3个水平
    orthogonal_table = [
        [1, 1, 1, 1],
        [1, 2, 2, 2],
        [1, 3, 3, 3],
        [2, 1, 2, 3],
        [2, 2, 3, 1],
        [2, 3, 1, 2],
        [3, 1, 3, 2],
        [3, 2, 1, 3],
        [3, 3, 2, 1]
    ]
    
    # 将正交表中的数字映射到实际参数值
    df = pd.DataFrame()
    
    # 处理前4个参数（正交表中的因素）
    for i, param in enumerate(['Tt', 'Ts', 'Tad', 'Twa']):
        if param in bounds:
            low, high = bounds[param]
            # 计算三个水平的值
            level1 = low
            level2 = (low + high) / 2
            level3 = high
            
            # 创建一个字典来存储每个水平对应的值
            level_map = {1: level1, 2: level2, 3: level3}
            
            # 将正交表中的数值转换为实际参数值
            column_values = [level_map[orthogonal_table[j][i]] for j in range(len(orthogonal_table))]
            df[param] = column_values
    
    # 添加固定的Ta值
    df['Ta'] = 0.0
    
    # 添加Tb参数值（使用中间值）
    tb_low, tb_high = bounds['Tb']
    df['Tb'] = (tb_low + tb_high) / 2
    
    # 重新排列列顺序以匹配原始顺序
    df = df[['Tt', 'Ts', 'Ta', 'Tad', 'Twa', 'Tb']]
    
    # 添加案例编号
    df.insert(0, 'case_id', range(1, len(df) + 1))
    
    return df

def main():
    print("正在生成简单的正交试验设计样本...")
    
    # 生成正交试验设计样本
    df = generate_simple_orthogonal()
    
    # 保存到CSV文件
    output_file = os.path.join("csv_data", "simple_orthogonal_samples.csv")
    df.to_csv(output_file, index=False)
    print(f"样本点已保存到: {output_file}")
    
    # 显示一些统计信息
    print("\n参数范围统计:")
    print(df.describe())
    
    # 显示前几行
    print("\n前5行数据:")
    print(df.head())
    
    print(f"\n总共生成了 {len(df)} 个参数组合")
    print("现在可以使用 run_all_cases.py 来运行所有案例")

if __name__ == "__main__":
    main()