$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '_python.ps1')
Push-Location $PSScriptRoot
try {
    & $cudaPython (Join-Path $PSScriptRoot 'selftest_v07.py') @args
    $resultCode = $LASTEXITCODE
} finally { Pop-Location }
exit $resultCode
