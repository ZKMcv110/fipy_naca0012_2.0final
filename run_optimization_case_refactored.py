#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
优化案例运行脚本（重构版本）
此脚本将:
1. 调用naca0012dat.py生成几何文件 (需要从命令行参数获取参数)
2. 生成网格
3. 运行求解器计算Nu和f
4. 输出结果
"""

import subprocess
import sys
import os
import argparse
import pickle
from typing import Tuple, Optional


class GeometryGenerator:
    """负责生成几何文件的类"""
    
    @staticmethod
    def run(Tt: float, Ts: float, Ta: float, Tad: float, Twa: float, Tb: float) -> bool:
        """
        调用naca0012dat.py生成几何文件
        
        Args:
            Tt: 横向间距系数
            Ts: 纵向间距系数
            Ta: 弯度系数
            Tad: 交错间距系数
            Twa: 宽长比
            Tb: 厚度系数
            
        Returns:
            是否成功执行
        """
        try:
            # 调用naca0012dat.py生成几何文件
            cmd = [
                "python", "naca0012dat.py",
                "--Tt", str(Tt),
                "--Ts", str(Ts),
                "--Ta", str(Ta),
                "--Tad", str(Tad),
                "--Twa", str(Twa),
                "--Tb", str(Tb)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("naca0012dat.py执行成功")
            if result.stdout:
                print(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            print(f"naca0012dat.py执行失败: {e}")
            if e.stderr:
                print(f"错误输出: {e.stderr}")
            return False
        except FileNotFoundError:
            print("未找到naca0012dat.py脚本")
            return False


class MeshGenerator:
    """负责生成网格的类"""
    
    @staticmethod
    def run() -> bool:
        """
        运行网格生成器
        
        Returns:
            是否成功执行
        """
        try:
            # 删除已存在的网格文件以强制重新生成
            mesh_file = "airfoil_array.msh2"
            if os.path.exists(mesh_file):
                os.remove(mesh_file)
                print(f"已删除旧网格文件: {mesh_file}")
            
            # 运行网格生成器
            cmd = ["python", "网格生成器2_三角形.py", "-nopopup"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("网格生成器执行成功")
            if result.stdout:
                print(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            print(f"网格生成器执行失败: {e}")
            if e.stderr:
                print(f"错误输出: {e.stderr}")
            return False
        except FileNotFoundError:
            print("未找到网格生成器脚本")
            return False


class SolverRunner:
    """负责运行求解器的类"""
    
    @staticmethod
    def run() -> Tuple[float, float]:
        """
        运行求解器计算Nu和f
        
        Returns:
            (Nu, f) 的元组
        """
        try:
            # 运行求解器
            cmd = ["python", "求解器_pvtnf.py"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("求解器执行成功")
            if result.stdout:
                print(result.stdout)
            
            # 从保存的结果文件中读取Nu和f
            solver_results_file = os.path.join('solver_results', 'latest_solution.pkl')
            if os.path.exists(solver_results_file):
                with open(solver_results_file, 'rb') as f:
                    results = pickle.load(f)
                    Nu = results.get('Nu', 0)
                    f_val = results.get('f', 0)
                    return Nu, f_val
            else:
                print("未找到求解器结果文件")
                return 0, 0
        except subprocess.CalledProcessError as e:
            print(f"求解器执行失败: {e}")
            if e.stderr:
                print(f"错误输出: {e.stderr}")
            return 0, 0
        except FileNotFoundError:
            print("未找到求解器脚本")
            return 0, 0


class ResultWriter:
    """负责写入结果的类"""
    
    @staticmethod
    def write_result(output_file: str, case_params: dict, Nu: float, f: float, 
                     target_param: float) -> None:
        """
        写入结果到文件
        
        Args:
            output_file: 输出文件路径
            case_params: 案例参数字典
            Nu: Nu值
            f: f值
            target_param: 目标参数值
        """
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 写入结果文件
        with open(output_file, 'w') as file_handle:
            file_handle.write(f"{case_params['case_id']},{case_params['Tt']},{case_params['Ts']},"
                    f"{case_params['Ta']},{case_params['Tad']},{case_params['Twa']},"
                    f"{case_params['Tb']},{Nu},{f},{target_param}\n")
        
        print(f"结果已写入: {output_file}")


def calculate_target_parameter(Nu: float, f: float) -> float:
    """
    计算目标参数 Nu/(f^(1/3))
    
    Args:
        Nu: Nu值
        f: f值
        
    Returns:
        目标参数值
    """
    if f > 0:
        target_param = Nu / (f**(1/3))
        return target_param
    else:
        return 0


def run_optimization_case(case_params: dict) -> int:
    """
    运行优化案例的主函数
    
    Args:
        case_params: 案例参数字典
        
    Returns:
        退出码
    """
    print(f"运行优化案例: Tt={case_params['Tt']}, Ts={case_params['Ts']}, "
          f"Ta={case_params['Ta']}, Tad={case_params['Tad']}, "
          f"Twa={case_params['Twa']}, Tb={case_params['Tb']}")
    
    # 1. 调用naca0012dat.py生成几何文件
    print("\n1. 生成几何文件...")
    if not GeometryGenerator.run(
        case_params['Tt'], case_params['Ts'], case_params['Ta'],
        case_params['Tad'], case_params['Twa'], case_params['Tb']
    ):
        print("几何文件生成失败，终止流程")
        return 1
    
    # 2. 生成网格
    print("\n2. 生成网格...")
    if not MeshGenerator.run():
        print("网格生成失败，终止流程")
        return 1
    
    # 3. 运行求解器
    print("\n3. 运行求解器...")
    Nu, f = SolverRunner.run()
    
    # 4. 计算目标参数
    print(f"\n4. 计算目标参数...")
    print(f"Nu: {Nu}")
    print(f"f: {f}")
    target_param = calculate_target_parameter(Nu, f)
    if target_param > 0:
        print(f"目标参数 Nu/(f^(1/3)): {target_param}")
    else:
        print("无法计算目标参数")
    
    # 5. 输出结果
    ResultWriter.write_result(
        case_params['output'], case_params, Nu, f, target_param
    )
    
    return 0


def parse_arguments() -> dict:
    """
    解析命令行参数
    
    Returns:
        参数字典
    """
    parser = argparse.ArgumentParser(description='运行优化案例')
    parser.add_argument('Tt', type=float, help='Tt 参数 (横向间距系数)')
    parser.add_argument('Ts', type=float, help='Ts 参数 (纵向间距系数)')
    parser.add_argument('Tad', type=float, help='Tad 参数 (交错间距系数)')
    parser.add_argument('Tb', type=float, help='Tb 参数 (厚度系数)')
    parser.add_argument('--Ta', type=float, default=0.0, help='Ta 参数 (弯度系数，默认为0.0)')
    parser.add_argument('--Twa', type=float, default=0.5, help='Twa 参数 (宽长比，默认为0.5)')
    parser.add_argument('--output', type=str, default=os.path.join('csv_data', 'result.csv'), 
                       help='结果输出文件')
    parser.add_argument('--case_id', type=float, default=0, help='案例ID')
    
    args = parser.parse_args()
    
    # 将参数转换为字典
    return {
        'Tt': args.Tt,
        'Ts': args.Ts,
        'Tad': args.Tad,
        'Tb': args.Tb,
        'Ta': args.Ta,
        'Twa': args.Twa,
        'output': args.output,
        'case_id': args.case_id
    }


def main():
    # 解析命令行参数
    case_params = parse_arguments()
    
    # 运行优化案例
    return run_optimization_case(case_params)


if __name__ == "__main__":
    sys.exit(main())