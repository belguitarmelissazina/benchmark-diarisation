# Experiment: pyannote-onnx-extended (full ONNX CPU pipeline)
# Runs on all 4 benchmark files (4 runs)
# Run from project root:  ./run_phase_pyannote_onnx.ps1
#
# This uses samson6460/pyannote-onnx-extended which is a pure ONNX Runtime
# implementation of pyannote speaker-diarization-3.1. No PyTorch needed.
# Models are downloaded from HuggingFace automatically on first run.

if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($Matches[1].Trim(), $Matches[2].Trim(), "Process")
        }
    }
    Write-Host "Loaded .env" -ForegroundColor DarkGray
}

$ErrorActionPreference = "Stop"
$py = "diarisation/Scripts/python.exe"

$files = @(
    @{ name = "AMI-IS";  audio = "benchmarks/ami/IS1009a.wav";
       ref = "benchmarks/ami/IS1009a.rttm";                          transcribe = $false },
    @{ name = "AMI-EN";  audio = "benchmarks/ami/EN2002c.wav";
       ref = "benchmarks/ami/EN2002c.rttm";                          transcribe = $false },
    @{ name = "FR-018";  audio = "benchmarks/summre/018a_EARZ/018a_EARZ.wav";
       ref = "benchmarks/summre/018a_EARZ/018a_EARZ.ref.rttm";       transcribe = $true },
    @{ name = "FR-069";  audio = "benchmarks/summre/069c_EEPL/069c_EEPL.wav";
       ref = "benchmarks/summre/069c_EEPL/069c_EEPL.ref.rttm";       transcribe = $true }
)

$totalRuns = $files.Count
$runNum = 0

Write-Host "============================================================" -ForegroundColor Yellow
Write-Host "  pyannote-onnx-extended: $totalRuns runs" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow

foreach ($f in $files) {
    $runNum++
    $runName = "$($f.name)_pyannote_onnx"
    Write-Host ""
    Write-Host "=== [$runNum/$totalRuns] $runName ===" -ForegroundColor Cyan

    $cmd = @(
        "-m", "diar_pipeline.run_pyannote_onnx",
        "-i", $f.audio,
        "--reference-rttm", $f.ref,
        "--experiment", "pyannote_onnx",
        "--run-name", $runName
    )
    if ($f.transcribe) { $cmd += "--transcribe" }

    & $py @cmd
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  pyannote-onnx experiment complete" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "In MLflow UI, compare pyannote_onnx experiment DERs against"
Write-Host "your best pipeline config from the phase ablations."
