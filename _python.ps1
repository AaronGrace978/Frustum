# Dot-sourced by the run_*.ps1 launchers. Sets $cudaPython.
# Order: FRUSTUM_PYTHON, then a .venv in this folder, then the original Codex environment.
$candidates = @(
    $env:FRUSTUM_PYTHON,
    (Join-Path $PSScriptRoot '.venv\Scripts\python.exe'),
    'C:\Users\AGrac\Documents\Codex\2026-11-13\new-chat-3\Frusium_loader\.venv-cuda\Scripts\python.exe'
)
$cudaPython = $null
foreach ($candidate in $candidates) {
    if ($candidate -and (Test-Path -LiteralPath $candidate)) { $cudaPython = $candidate; break }
}
if (-not $cudaPython) {
    throw 'CUDA Python environment missing. Set FRUSTUM_PYTHON to a python.exe with CUDA PyTorch, or create .venv here (see README).'
}
$env:PYTHONUTF8 = '1'
