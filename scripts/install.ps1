param([switch]$SkipModels)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
py -3.12 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.12 from python.org, then retry.' }
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -e '.[dev]'
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
if (-not $SkipModels) {
    & .\.venv\Scripts\audio-narration-setup.exe
    if ($LASTEXITCODE -ne 0) { throw 'Model setup failed; see the error above.' }
}
Write-Host 'Launch: .\.venv\Scripts\audio-narration.exe'
