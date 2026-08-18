# Phase 10 - Validate BEST_CONFIG on all 4 files (4 runs)
# Run from project root:  ./run_phase10.ps1
#
# Robustness check: re-run the winning config on every file. If any DER is worse
# than its Phase 1 baseline, the tuning was overfit - fall back to defaults.

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

# <<< EDIT: paste the ENTIRE winning config from Phases 2-9 >>>
$BestConfig = @(
    "--embed", "resnet34",
    "--estimate", "gmm_bic",
    "--cluster", "sc",
    "--vad-model", "silero",
    "--vad-threshold", "0.4"
    # Add any winning flags from Phase 6-9:
    # , "--no-enhance"
    # , "--win-len", "1.5", "--hop-len", "0.75"
    # , "--refine-vbx"
    # , "--silhouette-refine"
)

$totalRuns = $files.Count
$runNum = 0

foreach ($f in $files) {
    $runNum++
    $runName = "$($f.name)_best"
    Write-Host ""
    Write-Host "=== [$runNum/$totalRuns] $runName ===" -ForegroundColor Cyan

    $cmd = @("-m", "diar_pipeline.run", "-i", $f.audio, "--reference-rttm", $f.ref,
             "--experiment", "validate", "--run-name", $runName) + $BestConfig
    if ($f.transcribe) { $cmd += "--transcribe" }

    & $py @cmd
}

Write-Host ""
Write-Host "=== Phase 10 complete ===" -ForegroundColor Green
Write-Host "Compare each <name>_best DER against the corresponding Phase 1 baseline."
Write-Host "If any file is worse: tuning was overfit, revert to defaults."
