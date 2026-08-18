# Phase 13 - Min/max speaker range prior (4 runs)
# Run from project root:  ./run_phase13.ps1
#
# Only worth running if Phase 12 showed a large oracle-vs-best gap, meaning the
# estimator is guessing speaker counts wrong. A tight min/max prior often helps
# without hardcoding the exact count.

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

# Audio files + per-file (min, max) speaker prior centered on the true count
$files = @(
    @{ name = "AMI-IS";  audio = "benchmarks/ami/IS1009a.wav";
       ref = "benchmarks/ami/IS1009a.rttm";                          transcribe = $false; min = 3; max = 5 },
    @{ name = "AMI-EN";  audio = "benchmarks/ami/EN2002c.wav";
       ref = "benchmarks/ami/EN2002c.rttm";                          transcribe = $false; min = 2; max = 4 },
    @{ name = "FR-018";  audio = "benchmarks/summre/018a_EARZ/018a_EARZ.wav";
       ref = "benchmarks/summre/018a_EARZ/018a_EARZ.ref.rttm";       transcribe = $true;  min = 3; max = 5 },
    @{ name = "FR-069";  audio = "benchmarks/summre/069c_EEPL/069c_EEPL.wav";
       ref = "benchmarks/summre/069c_EEPL/069c_EEPL.ref.rttm";       transcribe = $true;  min = 3; max = 5 }
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
    $runName = "$($f.name)_range$($f.min)-$($f.max)"
    Write-Host ""
    Write-Host "=== [$runNum/$totalRuns] $runName ===" -ForegroundColor Cyan

    $cmd = @("-m", "diar_pipeline.run", "-i", $f.audio, "--reference-rttm", $f.ref,
             "--experiment", "range", "--run-name", $runName,
             "--min-speakers", "$($f.min)", "--max-speakers", "$($f.max)") + $BestConfig
    if ($f.transcribe) { $cmd += "--transcribe" }

    & $py @cmd
}

Write-Host ""
Write-Host "=== Phase 13 complete ===" -ForegroundColor Green
Write-Host "Compare each <name>_range against the corresponding _best (Phase 11) and _oracle (Phase 12)."
Write-Host "If close to oracle, the tight range is a good substitute for hardcoding."
