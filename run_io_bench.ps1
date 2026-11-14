$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '_python.ps1')
& $cudaPython (Join-Path $PSScriptRoot 'frusium_loader.py') bench-io @args
exit $LASTEXITCODE
