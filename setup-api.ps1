$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not [Environment]::Is64BitOperatingSystem) { throw '64-bit Windows is required.' }
if (-not (Get-Command py.exe -ErrorAction SilentlyContinue)) {
    throw 'Install Python 3.12 (64-bit) with the Python launcher: https://www.python.org/downloads/windows/'
}
& py -3.12 -c "import struct; assert struct.calcsize('P') == 8, '64-bit Python required'"
if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 (64-bit) is required.' }
if (-not (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue)) {
    throw 'Local speech recognition requires an NVIDIA CUDA GPU and a working NVIDIA driver. CPU-only, AMD and Intel GPU execution is not implemented.'
}
& nvidia-smi.exe --query-gpu=name,memory.total --format=csv,noheader
if ($LASTEXITCODE -ne 0) { throw 'NVIDIA driver check failed. Fix the driver before downloading the speech model.' }
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    & py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 environment setup failed.' }
}
& '.venv\Scripts\python.exe' -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& '.venv\Scripts\python.exe' -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency conflict detected.' }
# API translation needs only the local speech model. No Ollama or Qwen download.
& '.venv\Scripts\python.exe' bootstrap_models.py --only whisper
if ($LASTEXITCODE -ne 0) { throw 'Speech model setup failed. Rerun this script to resume downloads.' }
Write-Host 'Ready. Double-click Start API Captions.vbs and enter your own OpenAI API key in the app.'
