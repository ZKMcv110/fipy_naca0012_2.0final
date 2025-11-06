@echo off
echo 开始完整的优化流程...

echo 步骤 1: 生成参数样本
python generate_samples.py
if %errorlevel% neq 0 (
    echo 参数样本生成失败!
    pause
    exit /b %errorlevel%
)

echo 步骤 2: 运行所有优化案例
python run_all_cases_refactored.py
if %errorlevel% neq 0 (
    echo 优化案例运行失败!
    pause
    exit /b %errorlevel%
)

echo 优化流程完成! 结果保存在 csv_data/final_results.csv
pause