$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '_python.ps1')
& $cudaPython (Join-Path $PSScriptRoot 'benchmark_layers.py') --run @args
exit $LASTEXITCODE
