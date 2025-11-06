#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NACA0012翼型阵列流动和传热求解器
使用FiPy库求解Navier-Stokes方程和能量方程
"""

import numpy as np
import matplotlib
# 设置非交互式后端，避免在批量运行时弹出图形窗口
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fipy import *
from fipy.tools import numerix
from tqdm import tqdm
import os
from datetime import datetime
import pickle

# --- 检查核心库是否存在 ---
try:
    from scipy.interpolate import griddata
except ImportError:
    print("错误: 缺少 Scipy 库。")
    print("请使用 'pip install scipy' 或 'conda install scipy' 进行安装。")
    exit()


def load_mesh():
    """加载网格"""
    filename = "airfoil_array"
    msh_filename = f"{filename}.msh2"

    if not os.path.exists(msh_filename):
        print("-" * 50)
        print(f"\n错误: 未找到网格文件 '{msh_filename}'。")
        print("\n请先运行 'generate_mesh.py' 脚本来生成网格,")
        print("然后再重新运行此脚本。")
        print("-" * 50)
        exit()

    print(f"正在从 '{msh_filename}' 加载网格...")
    mesh = Gmsh2D(msh_filename)
    return mesh


def setup_physical_parameters():
    """设置物理参数"""
    # 流体属性
    mu = 0.1
    rho = 1.
    U = 10.

    # 热物理属性
    k_fluid = 2.0  # 流体导热系数
    Cp_fluid = 1.0  # 流体比热容
    T_inlet = 293.15  # 入口温度 (K, 例如 20°C)
    T_airfoil = 353.15  # 翼型表面恒定温度 (K, 例如 80°C)

    return mu, rho, U, k_fluid, Cp_fluid, T_inlet, T_airfoil


def setup_variables_and_boundary_conditions(mesh, U, T_inlet, T_airfoil):
    """设置变量和边界条件"""
    # 核心变量定义
    Vc = mesh.cellVolumes
    Vcf = CellVariable(mesh=mesh, value=Vc).faceValue

    Vx = CellVariable(mesh=mesh, name="x velocity", value=U)
    Vy = CellVariable(mesh=mesh, name="y velocity", value=0.)

    Vf = FaceVariable(mesh=mesh, rank=1)
    Vf.setValue((Vx.faceValue, Vy.faceValue))

    p = CellVariable(mesh=mesh, name="pressure", value=0.)
    pc = CellVariable(mesh=mesh, value=0.)
    apx = CellVariable(mesh=mesh, value=1.)

    # 温度场变量
    T = CellVariable(mesh=mesh, name="temperature", value=T_inlet)

    # 边界条件定义
    inletFace = mesh.physicalFaces["inlet"]
    outletFace = mesh.physicalFaces["outlet"]
    airfoilsFace = mesh.physicalFaces["airfoils"]
    top_bottomFace = mesh.physicalFaces["top"] | mesh.physicalFaces["bottom"]

    # 流场边界条件
    Vx.constrain(U, inletFace)
    Vy.constrain(0., inletFace)
    p.faceGrad.constrain(0., inletFace)
    pc.faceGrad.constrain(0., inletFace)

    Vx.faceGrad.constrain(0., outletFace)
    Vy.faceGrad.constrain(0., outletFace)
    p.constrain(0., outletFace)
    pc.constrain(0., outletFace)

    Vx.constrain(0., airfoilsFace)
    Vy.constrain(0., airfoilsFace)
    p.faceGrad.constrain(0., airfoilsFace)
    pc.faceGrad.constrain(0., airfoilsFace)

    Vx.faceGrad.constrain(0., top_bottomFace)
    Vy.faceGrad.constrain(0., top_bottomFace)
    p.constrain(0., top_bottomFace)
    pc.constrain(0., top_bottomFace)

    # 温度场边界条件
    T.constrain(T_inlet, inletFace)  # 入口温度恒定
    T.constrain(T_airfoil, airfoilsFace)  # 翼型表面温度恒定
    T.faceGrad.constrain(0, top_bottomFace)  # 上下壁面绝热
    T.faceGrad.constrain(0, outletFace)  # 出口自由流出

    return Vc, Vcf, Vx, Vy, Vf, p, pc, apx, T, inletFace, outletFace, airfoilsFace, top_bottomFace


def build_equations(rho, mu, k_fluid, Cp_fluid, Vf, Vx, Vy, p, apx, mesh, T, pc):
    """构建控制方程"""
    # 动量方程
    Vx_Eq = UpwindConvectionTerm(coeff=rho * Vf, var=Vx) == \
        DiffusionTerm(coeff=mu, var=Vx) - \
        ImplicitSourceTerm(coeff=1.0, var=p.grad[0])

    Vy_Eq = UpwindConvectionTerm(coeff=rho * Vf, var=Vy) == \
        DiffusionTerm(coeff=mu, var=Vy) - \
        ImplicitSourceTerm(coeff=1.0, var=p.grad[1])

    # 压力修正方程
    coeff = (1. / (apx.faceValue * mesh._faceAreas * mesh._cellDistances))
    # 添加稳定项
    coeff *= 0.5  # 减小系数以提高稳定性
    pc_Eq = DiffusionTerm(coeff=coeff, var=pc) - Vf.divergence == 0

    # 温度方程
    T_Eq = UpwindConvectionTerm(coeff=rho * Cp_fluid * Vf, var=T) == \
        DiffusionTerm(coeff=k_fluid, var=T)

    return Vx_Eq, Vy_Eq, pc_Eq, T_Eq


def overflow_prevention(Vx, Vy, p, V_limit=1e2, p_limit=2e3):
    """防止变量溢出"""
    Vx.value[Vx.value > V_limit] = V_limit
    Vx.value[Vx.value < -V_limit] = -V_limit
    Vy.value[Vy.value > V_limit] = V_limit
    Vy.value[Vy.value < -V_limit] = -V_limit
    p.value[p.value > p_limit] = p_limit
    p.value[p.value < -p_limit] = -V_limit


def sweep(Vx_Eq, Vy_Eq, pc_Eq, Vx, Vy, p, pc, Vf, Vc, Vcf, apx, Rp, Rv):
    """执行一次迭代"""
    overflow_prevention(Vx, Vy, p)

    Vx_Eq.cacheMatrix()
    xres = Vx_Eq.sweep(var=Vx, underRelaxation=Rv)
    xmat = Vx_Eq.matrix
    apx[:] = numerix.asarray(xmat.takeDiagonal())

    yres = Vy_Eq.sweep(var=Vy, underRelaxation=Rv)

    presgrad = p.grad
    facepresgrad = presgrad.faceValue
    Vf[0] = Vx.faceValue + Vcf / apx.faceValue * \
        (presgrad[0].faceValue - facepresgrad[0])
    Vf[1] = Vy.faceValue + Vcf / apx.faceValue * \
        (presgrad[1].faceValue - facepresgrad[1])

    pcres = pc_Eq.sweep(var=pc)

    p.setValue(p + Rp * pc)
    Vx.setValue(Vx - (Vc * pc.grad[0]) / apx)
    Vy.setValue(Vy - (Vc * pc.grad[1]) / apx)

    presgrad = p.grad
    facepresgrad = presgrad.faceValue
    Vf[0] = Vx.faceValue + Vcf / apx.faceValue * \
        (presgrad[0].faceValue - facepresgrad[0])
    Vf[1] = Vy.faceValue + Vcf / apx.faceValue * \
        (presgrad[1].faceValue - facepresgrad[1])

    return xres, yres, pcres


def value_range(val, a, b):
    """判断值是否在范围内"""
    return (val > a and val <= b)


def calculate_performance_parameters(mesh, p, T, Vf, rho, U, k_fluid, T_inlet, T_airfoil,
                                     inletFace, outletFace, airfoilsFace):
    """计算性能参数（摩擦系数和努塞尔数）"""
    print("\n开始进行后处理: 计算 Darcy 摩擦系数 (f) 和 Nusselt 数 (Nu)...")

    # 计算几何参数
    x_coords = mesh.cellCenters[0]  # x 坐标数组
    y_coords = mesh.cellCenters[1]  # y 坐标数组

    # 修复numerix函数调用
    L_channel = numerix.amax(x_coords.value) - numerix.amin(x_coords.value)  # 通道长度
    H_channel = numerix.amax(y_coords.value) - numerix.amin(y_coords.value)  # 通道高度
    D_h = 4.0 * (H_channel * L_channel) / (2.0 * (H_channel + L_channel))  # 水力直径

    # 计算摩擦系数 f
    # 获取入口和出口面上的压力值
    inlet_mask = x_coords.value < (numerix.amin(x_coords.value) + 1e-6)  # 入口边界条件
    outlet_mask = x_coords.value > (numerix.amax(x_coords.value) - 1e-6)  # 出口边界条件

    inlet_pressure_values = p.value[inlet_mask]
    outlet_pressure_values = p.value[outlet_mask]

    # 计算平均压力
    P_inlet_avg = numerix.average(inlet_pressure_values) if len(inlet_pressure_values) > 0 else 0.0
    P_outlet_avg = numerix.average(outlet_pressure_values) if len(outlet_pressure_values) > 0 else 0.0

    # 计算压降
    delta_P = P_inlet_avg - P_outlet_avg

    # 计算总体积流量
    Q_total = 0.0
    for face_vel in [Vx, Vy]:
        face_areas = face_vel.mesh._faceAreas
        face_normals = face_vel.mesh._orientedFaceNormals
        face_velocities = face_vel.arithmeticFaceValue.value
        # 只考虑入口处的流量
        inlet_face_mask = face_vel.mesh.facesLeft.value
        Q_total += numerix.sum(face_areas * face_normals[0] * face_velocities * inlet_face_mask)

    # 计算摩擦系数 (Darcy friction factor)
    A_c = H_channel * 1.0  # 假设单位深度的横截面积
    D_h = 2 * H_channel    # 对于二维平行板通道，水力直径为 2*H
    V_avg = Q_total / A_c  # 平均速度
    rho = 1.0              # 流体密度

    # Darcy 摩擦系数计算
    darcy_f = (2 * delta_P * D_h) / (L_channel * rho * V_avg**2) if V_avg != 0 else 0.0

    # 计算平均努塞尔数 Nu
    # 获取固体壁面(翼型)上的温度和热通量
    wetted_mask = T.mesh.physicalCells["airfoils"]  # 翼型表面单元
    if wetted_mask is not None and numerix.sum(wetted_mask) > 0:
        T_wall = T.value[wetted_mask]
        # 计算总散热量
        Q_total = numerix.sum(T.faceGrad.mag[..., wetted_mask].value * T.mesh._faceAreas[wetted_mask])
        # 翼型总湿面积
        A_wetted = numerix.sum(T.mesh._faceAreas[wetted_mask])
        # 参考温度差
        T_inlet = 293.15  # 入口温度
        T_outlet_avg = numerix.average(T.value[outlet_mask]) if len(outlet_mask) > 0 else T_inlet
        delta_T_lmtd = (T_outlet_avg - T_inlet) / numerix.log((T_outlet_avg + 1e-9) / (T_inlet + 1e-9))  # 对数平均温差
        # 计算平均传热系数和努塞尔数
        h_avg = Q_total / (A_wetted * delta_T_lmtd) if delta_T_lmtd != 0 else 0.0
        Nu_avg = h_avg * D_h  # 使用水力直径作为特征长度
    else:
        Q_total = 0.0
        A_wetted = 1.0
        T_outlet_avg = 293.15
        delta_T_lmtd = 1.0
        h_avg = 0.0
        Nu_avg = 0.0

    return darcy_f, Nu_avg, L_channel, H_channel, D_h, P_inlet_avg, P_outlet_avg, delta_P, Q_total, A_wetted, T_outlet_avg, delta_T_lmtd, h_avg


def save_solution(mesh, Vx, Vy, p, T, Nu_avg, darcy_f):
    """保存求解结果"""
    print("正在保存求解结果...")

    # 创建保存目录
    if not os.path.exists('solver_results'):
        os.makedirs('solver_results')

    # 准备要保存的数据
    results = {
        'mesh': mesh,
        'Vx': Vx,
        'Vy': Vy,
        'p': p,
        'T': T,
        'Nu': Nu_avg,
        'f': darcy_f,
        'target_param': Nu_avg / (darcy_f**(1/3)) if darcy_f != 0 else 0
    }

    # 保存结果
    with open('solver_results/latest_solution.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("求解结果已保存到 solver_results/latest_solution.pkl")


def extended_postprocessing(mesh, Vx, Vy, p, T, Nu_avg, darcy_f, L_channel, H_channel, D_h,
                            P_inlet_avg, P_outlet_avg, delta_P, Q_total, A_wetted, T_outlet_avg, delta_T_lmtd, h_avg):
    """
    扩展后处理: 生成更多可视化和数据输出
    """
    print("开始执行扩展后处理...")

    # 获取坐标
    x_coords = mesh.cellCenters[0]
    y_coords = mesh.cellCenters[1]

    # 修复numerix函数调用
    # 获取通道中心线上的数据 (y=0)
    center_line_indices = numerix.where(numerix.fabs(y_coords.value) < 0.5)[0]
    if len(center_line_indices) > 0:
        # 提取中心线数据
        x_center = x_coords.value[center_line_indices]
        p_center = p.value[center_line_indices]
        vx_center = Vx.value[center_line_indices]

        # 按x坐标排序
        sort_indices = numerix.argsort(x_center)
        x_center_sorted = x_center[sort_indices]
        p_center_sorted = p_center[sort_indices]
        vx_center_sorted = vx_center[sort_indices]

        # 保存中心线数据
        centerline_data = np.column_stack((x_center_sorted, p_center_sorted, vx_center_sorted))
        np.savetxt(os.path.join('solver_results', 'centerline_data.csv'), centerline_data,
                   delimiter=',', header='x,pressure,velocity_x', comments='')
        print("中心线数据已保存到 solver_results/centerline_data.csv")
    else:
        print("未找到中心线数据")

    # 计算和保存入口/出口剖面数据
    inlet_mask = x_coords.value < (numerix.amin(x_coords.value) + 1e-6)
    outlet_mask = x_coords.value > (numerix.amax(x_coords.value) - 1e-6)

    if numerix.sum(inlet_mask) > 0:
        y_inlet = y_coords.value[inlet_mask]
        vx_inlet = Vx.value[inlet_mask]
        vy_inlet = Vy.value[inlet_mask]
        T_inlet = T.value[inlet_mask]

        # 按y坐标排序
        sort_indices = numerix.argsort(y_inlet)
        y_inlet_sorted = y_inlet[sort_indices]
        vx_inlet_sorted = vx_inlet[sort_indices]
        vy_inlet_sorted = vy_inlet[sort_indices]
        T_inlet_sorted = T_inlet[sort_indices]

        inlet_profile_data = np.column_stack((y_inlet_sorted, vx_inlet_sorted, vy_inlet_sorted, T_inlet_sorted))
        np.savetxt(os.path.join('solver_results', 'inlet_profile.csv'), inlet_profile_data,
                   delimiter=',', header='y,velocity_x,velocity_y,temperature', comments='')
        print("入口剖面数据已保存到 solver_results/inlet_profile.csv")

    if numerix.sum(outlet_mask) > 0:
        y_outlet = y_coords.value[outlet_mask]
        vx_outlet = Vx.value[outlet_mask]
        vy_outlet = Vy.value[outlet_mask]
        T_outlet = T.value[outlet_mask]

        # 按y坐标排序
        sort_indices = numerix.argsort(y_outlet)
        y_outlet_sorted = y_outlet[sort_indices]
        vx_outlet_sorted = vx_outlet[sort_indices]
        vy_outlet_sorted = vy_outlet[sort_indices]
        T_outlet_sorted = T_outlet[sort_indices]

        outlet_profile_data = np.column_stack((y_outlet_sorted, vx_outlet_sorted, vy_outlet_sorted, T_outlet_sorted))
        np.savetxt(os.path.join('solver_results', 'outlet_profile.csv'), outlet_profile_data,
                   delimiter=',', header='y,velocity_x,velocity_y,temperature', comments='')
        print("出口剖面数据已保存到 solver_results/outlet_profile.csv")

    # 生成全局性能总结
    performance_summary = {
        "Nu": float(Nu_avg),
        "f": float(darcy_f),
        "L_channel": float(L_channel),
        "H_channel": float(H_channel),
        "D_h": float(D_h),
        "P_inlet_avg": float(P_inlet_avg),
        "P_outlet_avg": float(P_outlet_avg),
        "delta_P": float(delta_P),
        "Q_total": float(Q_total),
        "A_wetted": float(A_wetted),
        "T_outlet_avg": float(T_outlet_avg),
        "delta_T_lmtd": float(delta_T_lmtd),
        "h_avg": float(h_avg)
    }

    with open(os.path.join('solver_results', 'performance_summary.json'), 'w') as f:
        json.dump(performance_summary, f, indent=4)
    print("性能总结已保存到 solver_results/performance_summary.json")

    print("扩展后处理完成。")


def visualize_results(mesh, p, Vx, Vy, T, sum_res_list, MaxSweep, sum_res, T_inlet, T_airfoil, results_dir):
    """可视化结果"""
    if 'sum_res' in locals() and (np.isnan(sum_res) or np.isinf(sum_res)):
        print("\n计算发散，无法生成结果图。")
        # 移除了input()调用，避免程序等待用户输入
        return

    # 收敛历史图
    print("\n正在生成收敛历史图...")
    plt.figure()
    plt.plot(np.log10(np.array(sum_res_list)))
    plt.xlabel('sweep number')
    plt.ylabel('log10(sum of residuals)')
    plt.title('Convergence History')
    plt.grid()
    plt.savefig(os.path.join(results_dir, "convergence_history.png"))
    plt.close()

    print("\n正在生成场变量云图...")

    # 防止变量溢出
    overflow_prevention(Vx, Vy, p)

    # 压力场可视化
    plt.figure(figsize=(10, 8))
    viewer_p = Viewer(vars=p, title="Pressure Field")
    viewer_p.plot()
    plt.savefig(os.path.join(results_dir, "pressure_field.png"))
    plt.close()

    # X速度场可视化
    plt.figure(figsize=(10, 8))
    viewer_vx = Viewer(vars=Vx, title="X Velocity Field")
    viewer_vx.plot()
    plt.savefig(os.path.join(results_dir, "velocity_x.png"))
    plt.close()

    # Y速度场可视化
    plt.figure(figsize=(10, 8))
    viewer_vy = Viewer(vars=Vy, title="Y Velocity Field")
    viewer_vy.plot()
    plt.savefig(os.path.join(results_dir, "velocity_y.png"))
    plt.close()

    # 温度场可视化
    plt.figure(figsize=(10, 8))
    viewer_T = Viewer(vars=T, title="Temperature Field (K)",
                      datamin=T_inlet, datamax=T_airfoil)
    viewer_T.plot()
    plt.savefig(os.path.join(results_dir, "temperature_field.png"))
    plt.close()

    print("\n所有图片已保存。")
    if MaxSweep < 100:
        print(f"\n*** 调试运行成功 (MaxSweep={MaxSweep})！ ***")
        print("*** 后处理代码已无语法错误。 ***")
        print("*** 现在请将 MaxSweep 改回 300 进行正式计算。 ***")

    print("后处理完成。")


def main():
    """主函数"""
    # 加载网格
    mesh = load_mesh()

    # 设置物理参数
    mu, rho, U, k_fluid, Cp_fluid, T_inlet, T_airfoil = setup_physical_parameters()

    # 设置变量和边界条件
    Vc, Vcf, Vx, Vy, Vf, p, pc, apx, T, inletFace, outletFace, airfoilsFace, top_bottomFace = \
        setup_variables_and_boundary_conditions(mesh, U, T_inlet, T_airfoil)

    # 构建方程
    Vx_Eq, Vy_Eq, pc_Eq, T_Eq = build_equations(
        rho, mu, k_fluid, Cp_fluid, Vf, Vx, Vy, p, apx, mesh, T, pc)

    # 求解参数
    V_limit = 1e2
    p_limit = 2e3
    MaxSweep = 300  # 正式计算时使用300
    # MaxSweep = 5  # 调试时使用5
    res_limit = 1e-4
    sum_res_list = []
    sum_res = 1e10

    # 求解循环
    pbar = tqdm(range(MaxSweep), desc="求解流场 (SIMPLE)")
    for i in pbar:
       # 更平滑的松弛因子策略
        if sum_res > 1e4:
            Rp, Rv = 0.3, 0.6  # 初始阶段使用较小的松弛因子
        elif sum_res > 1e3:
            Rp, Rv = 0.4, 0.7
        elif sum_res > 1e2:
            Rp, Rv = 0.5, 0.8
        elif sum_res > 1e1:
            Rp, Rv = 0.6, 0.9
        else:
            Rp, Rv = 0.7, 0.95  # 接近收敛时使用较大的松弛因子

        xres, yres, pcres = sweep(
            Vx_Eq, Vy_Eq, pc_Eq, Vx, Vy, p, pc, Vf, Vc, Vcf, apx, Rp, Rv)
        sum_res = sum([abs(xres), abs(yres), abs(pcres)])
        sum_res_list.append(sum_res)
        pbar.set_postfix({"sum res": f'{sum_res:.2e}',
                         "Rp": f'{Rp:.2f}', "Rv": f'{Rv:.2f}'})

        if np.isnan(sum_res):
            print("\n错误: 计算出现无效值(NaN)，求解发散。")
            break
        if sum_res < res_limit and i > 5:  # 增加一个最小迭代次数
            print("\n残差收敛，流场求解完成。")
            break

    if i == MaxSweep - 1:
        print("\n达到最大迭代次数，流场可能未完全收敛。")

    # 求解温度场
    if 'sum_res' in locals() and not (np.isnan(sum_res) or np.isinf(sum_res)):
        print("\n开始求解温度场...")
        T_Eq.solve(var=T)
        print("温度场求解完成。")
    else:
        print("\n由于流场计算失败，跳过温度场求解。")

    # 创建统一的结果目录
    results_dir = os.path.join("results", datetime.now().strftime(
        "%Y%m%d_%H%M%S") + "_cfd_solution")
    os.makedirs(results_dir, exist_ok=True)

    # 后处理和性能参数计算
    if 'sum_res' in locals() and not (np.isnan(sum_res) or np.isinf(sum_res)):
        darcy_f, Nu_avg, L_channel, H_channel, D_h, P_inlet_avg, P_outlet_avg, delta_P, Q_total, A_wetted, T_outlet_avg, delta_T_lmtd, h_avg = calculate_performance_parameters(
            mesh, p, T, Vf, rho, U, k_fluid, T_inlet, T_airfoil,
            inletFace, outletFace, airfoilsFace)

        # 保存结果
        save_solution(mesh, Vx, Vy, p, T, Nu_avg, darcy_f)

        # 扩展后处理
        extended_postprocessing(mesh, Vx, Vy, p, T, Nu_avg, darcy_f, L_channel, H_channel, D_h,
                                P_inlet_avg, P_outlet_avg, delta_P, Q_total, A_wetted, T_outlet_avg,
                                delta_T_lmtd, h_avg, T_inlet, T_airfoil, results_dir)
    else:
        print("\n流场计算失败，跳过力与传热计算。")

    # 可视化
    visualize_results(mesh, p, Vx, Vy, T, sum_res_list,
                      MaxSweep, sum_res, T_inlet, T_airfoil, results_dir)


if __name__ == '__main__':
    main()