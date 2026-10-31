$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not [Environment]::Is64BitOperatingSystem) { throw '64-bit Windows is required.' }
if (-not (Get-Command py.exe -ErrorAction SilentlyContinue)) {
    throw 'Install Python 3.12 (64-bit) with the Python launcher: https://www.python.org/downloads/windows/'
}
& py -3.12 -c "import struct; assert struct.calcsize('P') == 8, '64-bit Python required'"
if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 (64-bit) is required.' }
# Team release (2026-10-31 user decision): API only on every PC, GPU or not.
# Captions = OpenAI Realtime, comments = GPT-6 Luna text. No CUDA, Whisper,
# speech model download or Ollama, even when an NVIDIA GPU is present.
$requirements = 'requirements-light.txt'
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    & py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 environment setup failed.' }
}
& '.venv\Scripts\python.exe' -m pip install -r $requirements
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& '.venv\Scripts\python.exe' -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency conflict detected.' }
Write-Host 'Ready (API only). Double-click Start API Captions.vbs and enter your OpenAI API key in the app.'
