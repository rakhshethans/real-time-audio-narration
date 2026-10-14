param(
    [switch]$SkipModels,
    [switch]$Launch,
    [ValidateSet('auto','cuda','cpu')][string]$Backend = 'auto'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
function Check-Step([string]$Message) {
    if ($LASTEXITCODE -ne 0) { throw $Message }
}
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    py -3.12 -m venv .venv
    Check-Step 'Install 64-bit Python 3.12 from python.org, then retry.'
}
$narrationPython = Join-Path (Get-Location) '.venv/Scripts/python.exe'
& $narrationPython -m pip install --upgrade pip
Check-Step 'Could not update pip.'
if ($Backend -eq 'auto') {
    $nvidiaFound = [bool](Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue)
    if (-not $nvidiaFound) {
        try { $nvidiaFound = [bool](Get-CimInstance Win32_VideoController |
                                   Where-Object Name -Match 'NVIDIA') } catch {}
    }
    $Backend = if ($nvidiaFound) { 'cuda' } else { 'cpu' }
}
if ($Backend -eq 'cuda') {
    $cudaInstalled = & $narrationPython -c "import importlib.metadata as m; print(m.version('torch') == '2.8.0+cu128')" 2>$null
    if ($cudaInstalled -ne 'True') {
        $narrationDrive = (Get-Item -LiteralPath (Get-Location).Path).PSDrive
        if ($narrationDrive.Free -lt 12GB) {
            throw 'CUDA setup needs at least 12 GiB free temporary disk space. Free space, then retry. Existing models are preserved.'
        }
    }
    Write-Host 'Installing NVIDIA CUDA 12.8 PyTorch (large one-time download)...'
    & $narrationPython -m pip install --upgrade 'torch==2.8.0+cu128' 'torchvision==0.23.0+cu128' --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir
} else {
    Write-Warning 'CPU mode cannot promise narration within a few seconds.'
    & $narrationPython -m pip install 'torch==2.8.0' 'torchvision==0.23.0' --index-url https://download.pytorch.org/whl/cpu
}
Check-Step 'PyTorch installation failed.'
& $narrationPython -m pip install -e '.[dev]'
Check-Step 'Application dependency installation failed.'
if ($Backend -eq 'cuda') {
    & $narrationPython -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable: update the NVIDIA driver and retry'; x=torch.ones(1, device='cuda'); torch.cuda.synchronize(); print('GPU ready:', torch.cuda.get_device_name(0))"
    Check-Step 'GPU verification failed. See the driver/CUDA error above.'
}
if (-not $SkipModels) {
    & $narrationPython -m audio_narration.setup_models
    Check-Step 'Model setup failed; retry to reuse completed downloads.'
}
Write-Host 'Installation complete. Start the camera only after opening the app.'
if ($Launch) { & $narrationPython -m audio_narration.main --device $Backend }
else { Write-Host 'Launch: .\.venv\Scripts\audio-narration.exe' }
