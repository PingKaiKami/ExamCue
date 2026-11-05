$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    $Python = Get-Command py -ErrorAction SilentlyContinue
    if ($Python) {
        & py -3 -m venv (Join-Path $ProjectRoot ".venv")
    }
    else {
        $Python = Get-Command python -ErrorAction SilentlyContinue
        if (-not $Python) {
            throw "找不到 Python 3.11+，請先安裝 Python。"
        }
        & python -m venv (Join-Path $ProjectRoot ".venv")
    }
}

& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -e ($ProjectRoot + "[dev]")

Write-Host ""
Write-Host "ExamCue 安裝完成。請執行 run_examcue.bat。" -ForegroundColor Green
Write-Host "若有 NVIDIA GPU，可執行 .\scripts\setup_cuda.ps1 安裝語音辨識加速套件。"

