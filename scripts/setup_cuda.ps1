$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "找不到 ExamCue 虛擬環境。請先執行 .\scripts\setup.ps1。"
}

& $VenvPython -m pip install -e ($ProjectRoot + "[cuda-windows]")

Write-Host ""
Write-Host "CUDA 12 cuBLAS 與 cuDNN 9 已安裝。" -ForegroundColor Green
Write-Host "請重新啟動 ExamCue，再到右上角齒輪執行 GPU 語音辨識測試。"
