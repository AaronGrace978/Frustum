$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '_python.ps1')
Push-Location $PSScriptRoot
try {
    & $cudaPython (Join-Path $PSScriptRoot 'expert_correctness.py') @args
    $resultCode = $LASTEXITCODE
} finally { Pop-Location }
exit $resultCode
