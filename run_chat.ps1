$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '_python.ps1')
& $cudaPython (Join-Path $PSScriptRoot 'frustum_chat.py') @args
exit $LASTEXITCODE
