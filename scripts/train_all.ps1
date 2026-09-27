param(
    [int]$Epochs = 0,
    [string]$Device = "0",
    [switch]$Resume
)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
Set-Location -LiteralPath $projectRoot

foreach ($model in @("fight", "garbage", "fallen_person", "mobile_phone", "accident")) {
    & $python -m machine_learning.pipeline validate --model $model
    $lastCheckpoint = Join-Path $projectRoot "machine_learning\runs\$model\weights\last.pt"
    if ($Resume -and (Test-Path -LiteralPath $lastCheckpoint)) {
        & $python -m machine_learning.pipeline train --model $model --device $Device --resume
    } elseif ($Epochs -gt 0) {
        & $python -m machine_learning.pipeline train --model $model --epochs $Epochs --device $Device
    } else {
        & $python -m machine_learning.pipeline train --model $model --device $Device
    }
    & $python -m machine_learning.pipeline evaluate --model $model --device $Device
    & $python -m machine_learning.pipeline export --model $model
}
