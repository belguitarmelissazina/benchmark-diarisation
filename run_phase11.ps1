# Phase 11 - Full pipeline with BEST_CONFIG on all 4 files (4 runs)
# Run from project root:  ./run_phase11.ps1
#
# Runs diarization + transcription + DER on all 4 files. Transcription is
# enabled for every file in this phase (we want the final ASR output too).

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
       ref = "benchmarks/ami/IS1009a.rttm" },
    @{ name = "AMI-EN";  audio = "benchmarks/ami/EN2002c.wav";
       ref = "benchmarks/ami/EN2002c.rttm" },
    @{ name = "FR-018";  audio = "benchmarks/summre/018a_EARZ/018a_EARZ.wav";
       ref = "benchmarks/summre/018a_EARZ/018a_EARZ.ref.rttm" },
    @{ name = "FR-069";  audio = "benchmarks/summre/069c_EEPL/069c_EEPL.wav";
       ref = "benchmarks/summre/069c_EEPL/069c_EEPL.ref.rttm" }
)

# <<< EDIT: paste the ENTIRE winning config from Phases 2-9 >>>
$BestConfig = @(
    "--embed", "resnet34",
    "--estimate", "gmm_bic",
    "--cluster", "sc",
    "--vad-model", "silero",
    "--vad-threshold", "0.4"
    # Add any winning flags from Phase 6-9 (see run_phase10.ps1 for examples)
)

$totalRuns = $files.Count
$runNum = 0

foreach ($f in $files) {
    $runNum++
    $runName = "$($f.name)_best"
    Write-Host ""
    Write-Host "=== [$runNum/$totalRuns] $runName ===" -ForegroundColor Cyan

    $cmd = @("-m", "diar_pipeline.run", "-i", $f.audio, "--reference-rttm", $f.ref,
             "--transcribe",
             "--experiment", "final", "--run-name", $runName) + $BestConfig

    & $py @cmd
}

Write-Host ""
Write-Host "=== Phase 11 complete ===" -ForegroundColor Green
Write-Host "Compare generated transcripts against the reference text where available."
