#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量运行所有优化案例（重构版本）

该脚本将:
1. 读取 csv_data/parameter_samples.csv 文件
2. 对每个案例调用 run_optimization_case.py
3. 收集结果并保存到 csv_data/final_results.csv
"""

import pandas as pd
import subprocess
import sys
import os
import time
import argparse
from typing import Tuple, Optional


class CaseRunner:
    """负责运行单个案例的类"""
    
    def __init__(self, timeout: int = 3600):
        """
        初始化案例运行器
        
        Args:
            timeout: 超时时间（秒）
        """
        self.timeout = timeout
        self.temp_result_file = "csv_data/temp_result.csv"
    
    def run_case(self, case_params: dict) -> str:
        """
        运行单个案例
        
        Args:
            case_params: 案例参数字典
            
        Returns:
            结果字符串
        """
        case_id = case_params['case_id']
        Tt = case_params['Tt']
        Ts = case_params['Ts']
        Ta = case_params['Ta']
        Tad = case_params['Tad']
        Twa = case_params['Twa']
        Tb = case_params['Tb']
        
        print(f"\n运行案例 {case_id}: Tt={Tt:.3f}, Ts={Ts:.3f}, Ta={Ta:.3f}, "
              f"Tad={Tad:.3f}, Twa={Twa:.3f}, Tb={Tb:.3f}")
        
        # 构建命令
        cmd = self._build_command(case_params)
        
        try:
            # 执行命令
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True, 
                timeout=self.timeout
            )
            print(f"案例 {case_id} 执行成功")
            
            # 读取结果
            return self._process_result(case_id, case_params, result)
            
        except subprocess.CalledProcessError as e:
            print(f"案例 {case_id} 执行失败: {e}")
            print(f"错误输出: {e.stderr}")
            print(f"标准输出: {e.stdout}")
            return self._handle_error(case_id, case_params, e)
            
        except subprocess.TimeoutExpired:
            print(f"案例 {case_id} 执行超时")
            return self._format_result(case_params, 0, 0, 0, "timeout")
    
    def _build_command(self, case_params: dict) -> list:
        """
        构建运行命令
        
        Args:
            case_params: 案例参数字典
            
        Returns:
            命令列表
        """
        return [
            "python", "run_optimization_case_refactored.py",
            str(case_params['Tt']), str(case_params['Ts']), 
            str(case_params['Tad']), str(case_params['Tb']),
            "--Ta", str(case_params['Ta']),
            "--Twa", str(case_params['Twa']),
            "--case_id", str(case_params['case_id']),
            "--output", self.temp_result_file
        ]
    
    def _process_result(self, case_id: float, case_params: dict, 
                       result: subprocess.CompletedProcess) -> str:
        """
        处理执行结果
        
        Args:
            case_id: 案例ID
            case_params: 案例参数
            result: 执行结果
            
        Returns:
            格式化的结果字符串
        """
        if os.path.exists(self.temp_result_file) and \
           os.path.getsize(self.temp_result_file) > 0:
            with open(self.temp_result_file, "r") as f:
                result_line = f.read().strip()
            os.remove(self.temp_result_file)  # 清理临时文件
            return result_line + ",success"
        else:
            print(f"案例 {case_id} 没有产生结果")
            return self._format_result(case_params, 0, 0, 0, "no_result")
    
    def _handle_error(self, case_id: float, case_params: dict, 
                     error: subprocess.CalledProcessError) -> str:
        """
        处理执行错误
        
        Args:
            case_id: 案例ID
            case_params: 案例参数
            error: 错误对象
            
        Returns:
            格式化的错误结果字符串
        """
        failure_reason = "execution_error"
        if "网格生成器执行失败" in error.stderr:
            failure_reason = "mesh_generation_failed"
        elif "naca0012dat.py执行失败" in error.stderr:
            failure_reason = "geometry_generation_failed"
        elif "求解器执行失败" in error.stderr:
            failure_reason = "solver_failed"
        return self._format_result(case_params, 0, 0, 0, failure_reason)
    
    def _format_result(self, case_params: dict, Nu: float, f: float, 
                      target_param: float, failure_reason: str) -> str:
        """
        格式化结果字符串
        
        Args:
            case_params: 案例参数
            Nu: Nu值
            f: f值
            target_param: 目标参数
            failure_reason: 失败原因
            
        Returns:
            格式化的结果字符串
        """
        return (f"{case_params['case_id']},{case_params['Tt']},{case_params['Ts']},"
                f"{case_params['Ta']},{case_params['Tad']},{case_params['Twa']},"
                f"{case_params['Tb']},{Nu},{f},{target_param},{failure_reason}")


class BatchRunner:
    """批量运行案例的类"""
    
    def __init__(self, sample_file: str = "csv_data/parameter_samples.csv", 
                 result_file: str = "csv_data/final_results.csv"):
        """
        初始化批量运行器
        
        Args:
            sample_file: 参数采样文件路径
            result_file: 结果文件路径
        """
        self.sample_file = sample_file
        self.result_file = result_file
        self.result_header = "case_id,Tt,Ts,Ta,Tad,Twa,Tb,Nu,f,target_param,failure_reason\n"
        self.case_runner = CaseRunner()
    
    def check_prerequisites(self) -> bool:
        """
        检查运行前提条件
        
        Returns:
            是否满足条件
        """
        if not os.path.exists(self.sample_file):
            print(f"错误: 找不到参数采样文件 '{self.sample_file}'")
            print("请先运行 generate_samples.py 生成采样点")
            return False
        return True
    
    def run_all_cases(self) -> int:
        """
        运行所有案例
        
        Returns:
            退出码
        """
        if not self.check_prerequisites():
            return 1
        
        # 读取参数采样文件
        print("读取参数采样文件...")
        df = pd.read_csv(self.sample_file)
        print(f"总共需要运行 {len(df)} 个案例")
        
        # 准备结果文件
        self._prepare_result_file()
        
        # 逐个运行案例
        start_time = time.time()
        for index, row in df.iterrows():
            self._run_single_case(index, row, len(df), start_time)
        
        total_time = time.time() - start_time
        self._print_summary(total_time)
        return 0
    
    def _prepare_result_file(self):
        """准备结果文件"""
        # 确保目录存在
        os.makedirs(os.path.dirname(self.result_file), exist_ok=True)
        
        # 写入头部
        with open(self.result_file, 'w') as f:
            f.write(self.result_header)
    
    def _run_single_case(self, index: int, row: pd.Series, total_cases: int, 
                        start_time: float):
        """
        运行单个案例
        
        Args:
            index: 索引
            row: 参数行
            total_cases: 总案例数
            start_time: 开始时间
        """
        # 转换行数据为参数字典
        case_params = {
            'case_id': row['case_id'],
            'Tt': row['Tt'],
            'Ts': row['Ts'],
            'Ta': row['Ta'],
            'Tad': row['Tad'],
            'Twa': row['Twa'],
            'Tb': row['Tb']
        }
        
        # 运行案例
        result_line = self.case_runner.run_case(case_params)
        
        # 保存结果
        with open(self.result_file, 'a') as f:
            f.write(result_line + '\n')
        
        # 显示进度
        self._print_progress(index, total_cases, start_time)
    
    def _print_progress(self, index: int, total_cases: int, start_time: float):
        """
        打印进度信息
        
        Args:
            index: 当前索引
            total_cases: 总案例数
            start_time: 开始时间
        """
        elapsed_time = time.time() - start_time
        avg_time_per_case = elapsed_time / (index + 1)
        remaining_cases = total_cases - (index + 1)
        estimated_remaining_time = avg_time_per_case * remaining_cases
        
        print(f"进度: {index + 1}/{total_cases} ({(index + 1)/total_cases*100:.1f}%)")
        print(f"预计剩余时间: {estimated_remaining_time/60:.1f} 分钟")
    
    def _print_summary(self, total_time: float):
        """
        打印总结信息
        
        Args:
            total_time: 总耗时（秒）
        """
        print(f"\n所有案例运行完成!")
        print(f"总耗时: {total_time/3600:.2f} 小时")
        print(f"结果已保存到: {self.result_file}")


def main():
    parser = argparse.ArgumentParser(description='批量运行所有优化案例')
    parser.add_argument('--sample-file', type=str, 
                       default="csv_data/parameter_samples.csv",
                       help='参数采样文件路径')
    parser.add_argument('--result-file', type=str,
                       default="csv_data/final_results.csv",
                       help='结果文件路径')
    parser.add_argument('--timeout', type=int, default=3600,
                       help='单个案例超时时间（秒）')
    
    args = parser.parse_args()
    
    # 更新CaseRunner的超时设置
    runner = BatchRunner(args.sample_file, args.result_file)
    runner.case_runner.timeout = args.timeout
    
    return runner.run_all_cases()


if __name__ == "__main__":
    sys.exit(main())