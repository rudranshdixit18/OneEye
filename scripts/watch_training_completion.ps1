[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [int]$TrainerProcessId,

    [Parameter(Mandatory = $true)]
    [ValidateSet("fight", "garbage", "fallen_person", "mobile_phone", "accident")]
    [string]$Model,

    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = "Stop"
$resolvedRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$python = Join-Path $resolvedRoot ".venv\Scripts\python.exe"
$report = Join-Path $resolvedRoot "machine_learning\reports\${Model}_metrics.json"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Project Python is missing: $python"
}

Add-Type -TypeDefinition @"
using System.Runtime.InteropServices;
public static class OneEyeSleepControl {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint flags);
}
"@

$continuous = [uint32]2147483648
$systemRequired = [uint32]1
$awakeResult = [OneEyeSleepControl]::SetThreadExecutionState($continuous -bor $systemRequired)
if ($awakeResult -eq 0) {
    throw "Windows rejected the temporary keep-awake request."
}

try {
    $trainer = Get-Process -Id $TrainerProcessId -ErrorAction Stop
    Write-Output "Waiting for $Model trainer PID $TrainerProcessId; system idle sleep is suppressed."
    $trainer.WaitForExit()
    $trainingStatusPath = Join-Path $resolvedRoot "machine_learning\training_status.json"
    $trainingStatus = (Get-Content -LiteralPath $trainingStatusPath -Raw | ConvertFrom-Json).models.$Model
    if ($trainingStatus.stage -ne "train" -or $trainingStatus.status -ne "complete") {
        throw "$Model process exited without a durable successful-training marker; evaluation was not started."
    }

    Set-Location -LiteralPath $resolvedRoot
    & $python -m machine_learning.pipeline evaluate --model $Model --device 0
    if ($LASTEXITCODE -ne 0) {
        throw "$Model untouched-test evaluation failed with exit code $LASTEXITCODE."
    }

    $metrics = Get-Content -LiteralPath $report -Raw | ConvertFrom-Json
    if ($metrics.quality_gate.passed -eq $true) {
        & $python -m machine_learning.pipeline export --model $Model
        if ($LASTEXITCODE -ne 0) {
            throw "$Model passed evaluation but export failed with exit code $LASTEXITCODE."
        }
        Write-Output "$Model passed the strict quality gate and was exported."
    } else {
        Write-Output "$Model did not pass the strict quality gate; the research checkpoint was not promoted."
    }
} finally {
    [void][OneEyeSleepControl]::SetThreadExecutionState($continuous)
}
