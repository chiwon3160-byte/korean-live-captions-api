$ErrorActionPreference = 'Stop'
try {
    $captionRoot = $PSScriptRoot
    $captionPython = Join-Path $captionRoot '.venv\Scripts\pythonw.exe'
    if (-not (Test-Path -LiteralPath $captionPython)) {
        throw 'Python environment missing. Run setup-api.ps1 first.'
    }
    $captionEntry = Join-Path $captionRoot 'run_app.py'
    if (-not (Test-Path -LiteralPath $captionEntry)) {
        throw 'App files are missing. Extract the complete ZIP into a local folder.'
    }
    Start-Process -FilePath $captionPython -ArgumentList ('"' + $captionEntry + '" --api') -WorkingDirectory $captionRoot
} catch {
    Add-Type -AssemblyName System.Windows.Forms
    [System.Windows.Forms.MessageBox]::Show($_.Exception.Message, 'Caption startup error', 'OK', 'Error') | Out-Null
}
