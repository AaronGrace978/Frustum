$ErrorActionPreference = 'Stop'
$cudaPython = $env:FRUSIUM_PYTHON
if (!$cudaPython) {
    $cudaPython = 'C:\Users\AGrac\Documents\Codex\2026-11-13\new-chat-3\Frusium_loader\.venv-cuda\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $cudaPython)) { throw 'CUDA Python environment missing. Set FRUSIUM_PYTHON.' }
$env:PYTHONUTF8 = '1'
& $cudaPython (Join-Path $PSScriptRoot 'benchmark_glm_placement.py') @args
exit $LASTEXITCODE
