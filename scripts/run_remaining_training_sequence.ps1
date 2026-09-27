[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [int]$InitialTrainerProcessId,

    [string]$InitialModel = "garbage",

    [string[]]$RemainingModels = @("fallen_person", "mobile_phone", "accident"),

    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = "Stop"
$resolvedRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$python = Join-Path $resolvedRoot ".venv\Scripts\python.exe"
$runsRoot = (Resolve-Path -LiteralPath (Join-Path $resolvedRoot "machine_learning\runs")).Path
$allowedModels = @("fight", "garbage", "fallen_person", "mobile_phone", "accident")

if (-not (Test-Path -LiteralPath $python)) {
    throw "Project Python is missing: $python"
}
foreach ($model in @($InitialModel) + $RemainingModels) {
    if ($model -notin $allowedModels) {
        throw "Unsupported model in training sequence: $model"
    }
}

Add-Type -TypeDefinition @"
using System.Runtime.InteropServices;
public static class OneEyeSequenceSleepControl {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint flags);
}
"@

$continuous = [uint32]2147483648
$systemRequired = [uint32]1
$awakeResult = [OneEyeSequenceSleepControl]::SetThreadExecutionState($continuous -bor $systemRequired)
if ($awakeResult -eq 0) {
    throw "Windows rejected the temporary keep-awake request."
}

function Invoke-Pipeline {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    & $python -m machine_learning.pipeline @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Pipeline command failed ($LASTEXITCODE): $($Arguments -join ' ')"
    }
}

function Complete-Model {
    param([Parameter(Mandatory = $true)][string]$Model)
    Invoke-Pipeline -Arguments @("evaluate", "--model", $Model, "--device", "0")
    $reportPath = Join-Path $resolvedRoot "machine_learning\reports\${Model}_metrics.json"
    $metrics = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
    if ($metrics.quality_gate.passed -eq $true) {
        Invoke-Pipeline -Arguments @("export", "--model", $Model)
        Write-Output "$Model passed the strict quality gate and was exported."
    } else {
        Write-Output "$Model did not pass the strict quality gate; its research run was preserved."
    }
}

function Archive-ExistingRun {
    param([Parameter(Mandatory = $true)][string]$Model)
    $sourceCandidate = Join-Path $runsRoot $Model
    if (-not (Test-Path -LiteralPath $sourceCandidate)) {
        return
    }
    $source = (Resolve-Path -LiteralPath $sourceCandidate).Path
    $runsPrefix = $runsRoot.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    if (-not $source.StartsWith($runsPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to archive a run outside the runs directory: $source"
    }
    $suffix = Get-Date -Format "yyyyMMdd-HHmmss"
    $target = Join-Path $runsRoot "${Model}_previous_${suffix}"
    if (Test-Path -LiteralPath $target) {
        throw "Run archive target already exists: $target"
    }
    Move-Item -LiteralPath $source -Destination $target
    Write-Output "Preserved previous $Model run at $target"
}

try {
    Set-Location -LiteralPath $resolvedRoot
    $trainer = Get-Process -Id $InitialTrainerProcessId -ErrorAction Stop
    Write-Output "Waiting for $InitialModel trainer PID $InitialTrainerProcessId; idle sleep is suppressed."
    $trainer.WaitForExit()
    $trainingStatusPath = Join-Path $resolvedRoot "machine_learning\training_status.json"
    $initialStatus = (Get-Content -LiteralPath $trainingStatusPath -Raw | ConvertFrom-Json).models.$InitialModel
    if ($initialStatus.stage -ne "train" -or $initialStatus.status -ne "complete") {
        throw "$InitialModel process exited without a durable successful-training marker."
    }
    Complete-Model -Model $InitialModel

    foreach ($model in $RemainingModels) {
        Archive-ExistingRun -Model $model
        Invoke-Pipeline -Arguments @("train", "--model", $model, "--device", "0")
        Complete-Model -Model $model
    }
    Write-Output "The configured OneEye training sequence completed."
} finally {
    [void][OneEyeSequenceSleepControl]::SetThreadExecutionState($continuous)
}
