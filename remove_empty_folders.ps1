# 删除 results 目录下所有空文件夹的 PowerShell 脚本

$ResultsPath = "f:\pyProject\fipy_naca0012_2.0\results"

Write-Host "正在搜索空文件夹..." -ForegroundColor Yellow

# 获取所有空文件夹
$EmptyFolders = Get-ChildItem -Path $ResultsPath -Directory -Recurse | Where-Object { (Get-ChildItem $_.FullName -Force | Measure-Object).Count -eq 0 }

if ($EmptyFolders.Count -eq 0) {
    Write-Host "未找到空文件夹。" -ForegroundColor Green
    exit
}

Write-Host "找到 $($EmptyFolders.Count) 个空文件夹：" -ForegroundColor Yellow

foreach ($Folder in $EmptyFolders) {
    Write-Host "删除: $($Folder.FullName)"
    Remove-Item -Path $Folder.FullName -Force
}

Write-Host "已完成删除所有空文件夹。" -ForegroundColor Green
