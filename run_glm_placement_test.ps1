$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '_python.ps1')
& $cudaPython (Join-Path $PSScriptRoot 'benchmark_glm_placement.py') @args
exit $LASTEXITCODE
