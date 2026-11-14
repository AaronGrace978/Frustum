$ErrorActionPreference = 'Stop'
$cudaPython = 'C:\Users\AGrac\Documents\Codex\2026-11-13\new-chat-3\Frusium_loader\.venv-cuda\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $cudaPython)) { throw 'CUDA Python environment missing' }
Push-Location $PSScriptRoot
try {
    & $cudaPython (Join-Path $PSScriptRoot 'expert_correctness.py') @args
    $resultCode = $LASTEXITCODE
} finally { Pop-Location }
exit $resultCode
