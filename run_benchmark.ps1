$ErrorActionPreference = 'Stop'
$cudaPython = 'C:\Users\AGrac\Documents\Codex\2026-11-13\new-chat-3\Frusium_loader\.venv-cuda\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $cudaPython)) { throw 'CUDA environment missing. Recreate the isolated CUDA PyTorch environment.' }
& $cudaPython (Join-Path $PSScriptRoot 'benchmark_layers.py') --run @args
exit $LASTEXITCODE
