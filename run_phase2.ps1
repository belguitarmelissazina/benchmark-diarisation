# Phase 2 - VAD ablation: threshold sweep + pyannote (28 runs, ~2h30)
# Run from project root:  ./run_phase2.ps1

# Load .env file (HF_TOKEN etc.)
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

# Audio files
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

# Phase 2A: Silero threshold sweep
$thresholds = @(0.30, 0.35, 0.40, 0.45, 0.55, 0.65)
$sileroRuns = $thresholds.Count * $files.Count
$totalRuns  = $sileroRuns + $files.Count

Write-Host "============================================================" -ForegroundColor Yellow
Write-Host "  Phase 2A - Silero sweep: $sileroRuns runs" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow

$runNum = 0

foreach ($th in $thresholds) {
    foreach ($f in $files) {
        $runNum++
        $thStr = "$th".Replace(".", "")
        $runName = "$($f.name)_silero_th$thStr"
        Write-Host ""
        Write-Host "=== [$runNum/$totalRuns] $runName ===" -ForegroundColor Cyan

        $cmd = @(
            "-m", "diar_pipeline.run",
            "-i", $f.audio,
            "--reference-rttm", $f.ref,
            "--experiment", "ablate_vad",
            "--run-name", $runName,
            "--vad-model", "silero",
            "--vad-threshold", "$th"
        )
        if ($f.transcribe) { $cmd += "--transcribe" }

        & $py @cmd
    }
}

# Phase 2B: pyannote segmentation-3.0
Write-Host ""
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host "  Phase 2B - pyannote VAD: $($files.Count) runs" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow

if (-not $env:HF_TOKEN) {
    Write-Host ""
    Write-Host "WARNING: HF_TOKEN not set. Skipping pyannote runs." -ForegroundColor Red
    Write-Host 'Set it in .env or run: $env:HF_TOKEN = "hf_xxxxx"' -ForegroundColor Red
} else {
    foreach ($f in $files) {
        $runNum++
        $runName = "$($f.name)_pyannote"
        Write-Host ""
        Write-Host "=== [$runNum/$totalRuns] $runName ===" -ForegroundColor Cyan

        $cmd = @(
            "-m", "diar_pipeline.run",
            "-i", $f.audio,
            "--reference-rttm", $f.ref,
            "--experiment", "ablate_vad",
            "--run-name", $runName,
            "--vad-model", "pyannote",
            "--vad-threshold", "0.50"
        )
        if ($f.transcribe) { $cmd += "--transcribe" }

        & $py @cmd
    }
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Phase 2 complete" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "In MLflow UI, open ablate_vad experiment."
Write-Host "Sort by der. Compare vad_model and vad_threshold columns."
