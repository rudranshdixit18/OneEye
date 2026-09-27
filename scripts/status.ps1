$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
Set-Location -LiteralPath $projectRoot

Write-Host "OneEye implementation status" -ForegroundColor Cyan
git log -3 --oneline

if (Test-Path -LiteralPath $python) {
    & $python -c "import torch; print('PyTorch', torch.__version__, '| CUDA=', torch.cuda.is_available(), '| Device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
}

Get-Content -LiteralPath "machine_learning\training_status.json"

foreach ($model in @("fight", "garbage", "fallen_person", "mobile_phone", "accident")) {
    $best = Join-Path $projectRoot "models\$model\best.pt"
    $last = Join-Path $projectRoot "machine_learning\runs\$model\weights\last.pt"
    if (Test-Path -LiteralPath $best) {
        Write-Host "$model : exported" -ForegroundColor Green
    } elseif (Test-Path -LiteralPath $last) {
        Write-Host "$model : resumable training checkpoint" -ForegroundColor Yellow
    } else {
        Write-Host "$model : no fresh training checkpoint" -ForegroundColor DarkYellow
    }
}
