# Phase 12 - Oracle / sanity bounds (4 runs)
# Run from project root:  ./run_phase12.ps1
#
# Forces the true speaker count per file to see the pipeline's floor. The gap
# between oracle and best (Phase 10/11) tells you how much DER comes from
# speaker-count errors vs everything else.

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

# Audio files + per-file oracle speaker count
$files = @(
    @{ name = "AMI-IS";  audio = "benchmarks/ami/IS1009a.wav";
       ref = "benchmarks/ami/IS1009a.rttm";                          transcribe = $false; nspk = 4 },
    @{ name = "AMI-EN";  audio = "benchmarks/ami/EN2002c.wav";
       ref = "benchmarks/ami/EN2002c.rttm";                          transcribe = $false; nspk = 3 },
    @{ name = "FR-018";  audio = "benchmarks/summre/018a_EARZ/018a_EARZ.wav";
       ref = "benchmarks/summre/018a_EARZ/018a_EARZ.ref.rttm";       transcribe = $true;  nspk = 4 },
    @{ name = "FR-069";  audio = "benchmarks/summre/069c_EEPL/069c_EEPL.wav";
       ref = "benchmarks/summre/069c_EEPL/069c_EEPL.ref.rttm";       transcribe = $true;  nspk = 4 }
)

# <<< EDIT: paste the ENTIRE winning config from Phases 2-9 >>>
$BestConfig = @(
    "--embed", "resnet34",
    "--estimate", "gmm_bic",
    "--cluster", "sc",
    "--vad-model", "silero",
    "--vad-threshold", "0.4"
)

$totalRuns = $files.Count
$runNum = 0

foreach ($f in $files) {
    $runNum++
    $runName = "$($f.name)_oracle$($f.nspk)"
    Write-Host ""
    Write-Host "=== [$runNum/$totalRuns] $runName ===" -ForegroundColor Cyan

    $cmd = @("-m", "diar_pipeline.run", "-i", $f.audio, "--reference-rttm", $f.ref,
             "--experiment", "oracle", "--run-name", $runName,
             "--num-speakers", "$($f.nspk)") + $BestConfig
    if ($f.transcribe) { $cmd += "--transcribe" }

    & $py @cmd
}

Write-Host ""
Write-Host "=== Phase 12 complete ===" -ForegroundColor Green
Write-Host "Read: (oracle DER - best DER) = DER attributable to speaker-count errors."
