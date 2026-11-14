$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '_python.ps1')
Push-Location $PSScriptRoot
try {
    & $cudaPython (Join-Path $PSScriptRoot 'reference_parity.py') @args
    $resultCode = $LASTEXITCODE
} finally { Pop-Location }
exit $resultCode
