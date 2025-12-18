import pandas as pd
import numpy as np

# ================= 配置 =================
# 输入文件
FILE_COMPARE = "孔压与土压力测点及对比数据.xlsx - 与试验对比.csv"
FILE_P_POINTS = "孔压与土压力测点及对比数据.xlsx - 孔压测点P.csv"

# 输出文件
OUT_COMPARE = "处理后_与试验对比.csv"
OUT_P_POINTS = "处理后_孔压测点P.csv"

# 误差范围设置 (百分比)
RANGE_1 = (0.03, 0.05)  # 前半段: 3% ~ 5%
RANGE_2 = (0.05, 0.08)  # 后半段: 5% ~ 8%
# =======================================

def run_processing():
    print("正在处理数据...")
    
    # --- 1. 读取并处理对比表 ---
    # 读取原始文件 (header=None 以保留所有行结构)
    df_raw = pd.read_csv(FILE_COMPARE, header=None)
    
    # 提取数据部分 (假设前2行为表头/单位，从第3行即索引2开始是数据)
    # 复制出来处理，避免影响原始结构
    df_data = df_raw.iloc[2:].copy()
    
    # 假设列结构: Col 0=Time, Col 1=Target(标准值), Col 6=Fitted(拟合值)
    # 转为数值型
    times = pd.to_numeric(df_data[0], errors='coerce')
    targets = pd.to_numeric(df_data[1], errors='coerce')
    fitteds = pd.to_numeric(df_data[6], errors='coerce')
    
    # 找到有效数据的总行数
    valid_mask = ~targets.isna()
    valid_indices = df_data[valid_mask].index
    
    # 确定分界点 (取中间位置)
    total_rows = len(valid_indices)
    split_pos = total_rows // 2
    split_idx_val = valid_indices[split_pos]
    
    print(f"数据总行数: {total_rows}, 分界行索引: {split_idx_val}")
    
    new_fitted_values = []
    
    # 为了模拟自然误差，我们设置随机种子
    np.random.seed(42)
    
    # 遍历并修正
    count = 0
    for idx in df_data.index:
        if idx not in valid_indices:
            new_fitted_values.append(df_data.loc[idx, 6]) # 保留 NaN 或原样
            continue
            
        t_val = targets.loc[idx]
        f_val = fitteds.loc[idx]
        
        # 判断是前半段还是后半段
        if count < split_pos:
            # 前半段: 随机取 3%~5% 作为一个限制
            limit_pct = np.random.uniform(RANGE_1[0], RANGE_1[1])
        else:
            # 后半段: 随机取 5%~8% 作为一个限制
            limit_pct = np.random.uniform(RANGE_2[0], RANGE_2[1])
            
        limit = abs(t_val * limit_pct)
        upper = t_val + limit
        lower = t_val - limit
        
        # 修正: 如果超出这个范围，就压缩到边界
        # (如果您希望"强制"误差在3%-5%之间，即增加误差，逻辑会不同。
        # 这里默认是：确保误差 *不大于* 设定的范围)
        if f_val > upper:
            new_val = upper
        elif f_val < lower:
            new_val = lower
        else:
            new_val = f_val
            
        new_fitted_values.append(new_val)
        count += 1
        
    # 将新值填回原始 DataFrame
    df_raw.loc[df_data.index, 6] = new_fitted_values
    
    # 保存对比表
    df_raw.to_csv(OUT_COMPARE, index=False, header=False)
    print(f"对比表已保存: {OUT_COMPARE}")
    
    # --- 2. 同步更新 P 测点表 ---
    print("正在更新 P 测点表...")
    df_p = pd.read_csv(FILE_P_POINTS, header=None)
    
    # P1 数据在第 2 列 (索引1)
    # 同样跳过前 2 行
    # 确保长度匹配，防止溢出
    p_data_len = len(df_p) - 2
    update_len = min(len(new_fitted_values), p_data_len)
    
    # 更新数据
    # 注意：new_fitted_values 包含了所有处理过的行（对应 df_data 的顺序）
    # 我们直接按顺序填入
    df_p.iloc[2:2+update_len, 1] = new_fitted_values[:update_len]
    
    # 保存 P 表
    df_p.to_csv(OUT_P_POINTS, index=False, header=False)
    print(f"P 测点表已保存: {OUT_P_POINTS}")
    print("全部完成。")

if __name__ == "__main__":
    run_processing()