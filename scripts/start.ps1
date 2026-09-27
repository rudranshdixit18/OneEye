$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Run scripts\setup.ps1 first."
}
Set-Location -LiteralPath $projectRoot
$envFile = Join-Path $projectRoot ".env"
$hostAddress = "127.0.0.1"
$port = "8000"
if (Test-Path -LiteralPath $envFile) {
    foreach ($line in Get-Content -LiteralPath $envFile) {
        if ($line -match '^\s*ONEEYE_HOST\s*=\s*(.+?)\s*$') {
            $hostAddress = $Matches[1].Trim('"', "'")
        } elseif ($line -match '^\s*ONEEYE_PORT\s*=\s*(\d+)\s*$') {
            $port = $Matches[1]
        }
    }
    & $python -m uvicorn backend.app.main:app --env-file $envFile --host $hostAddress --port $port
} else {
    & $python -m uvicorn backend.app.main:app --host $hostAddress --port $port
}
