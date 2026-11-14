$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '_python.ps1')
& $cudaPython (Join-Path $PSScriptRoot 'phi4_chat.py') @args
exit $LASTEXITCODE
