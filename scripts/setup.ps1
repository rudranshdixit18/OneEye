$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

Set-Location -LiteralPath $projectRoot
if (-not (Test-Path -LiteralPath $venvPython)) {
    $pythonLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($pythonLauncher) {
        & $pythonLauncher.Source -3.11 -m venv .venv
        if ($LASTEXITCODE -ne 0) {
            & $pythonLauncher.Source -3 -m venv .venv
        }
    } else {
        $systemPython = Get-Command python.exe -ErrorAction SilentlyContinue
        if (-not $systemPython) {
            throw "Python 3.11 or newer is required. Install Python, then run this script again."
        }
        & $systemPython.Source -m venv .venv
    }
    if (-not (Test-Path -LiteralPath $venvPython)) {
        throw "Unable to create .venv with an available Python 3 installation."
    }
}
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r backend\requirements-dev.txt
& $venvPython -m pip install -r machine_learning\requirements.txt
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    & $venvPython -m pip install --upgrade torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
}
if (-not (Test-Path -LiteralPath (Join-Path $projectRoot ".env"))) {
    Copy-Item -LiteralPath (Join-Path $projectRoot ".env.example") -Destination (Join-Path $projectRoot ".env")
}
Write-Host "OneEye environment is ready." -ForegroundColor Green
