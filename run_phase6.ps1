# Phase 6 - Similarity enhancement ablation (2 settings x 4 files = 8 runs, SC only)
# Run from project root:  ./run_phase6.ps1
# Skip this phase if Phase 5 did not pick spectral clustering (sc).

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

# <<< EDIT after Phases 3-5 >>>
$BestEmbed    = "resnet34"
$BestEstimate = "gmm_bic"
$BestVad      = @("--vad-model", "silero", "--vad-threshold", "0.4")

$totalRuns = 2 * $files.Count
$runNum = 0

foreach ($f in $files) {
    # 6.1 Enhancement ON
    $runNum++
    $runName = "$($f.name)_enhance_on"
    Write-Host ""
    Write-Host "=== [$runNum/$totalRuns] $runName ===" -ForegroundColor Cyan
    $cmd = @("-m", "diar_pipeline.run", "-i", $f.audio, "--reference-rttm", $f.ref,
             "--experiment", "ablate_enhance", "--run-name", $runName,
             "--embed", $BestEmbed, "--estimate", $BestEstimate,
             "--cluster", "sc") + $BestVad
    if ($f.transcribe) { $cmd += "--transcribe" }
    & $py @cmd

    # 6.2 Enhancement OFF
    $runNum++
    $runName = "$($f.name)_enhance_off"
    Write-Host ""
    Write-Host "=== [$runNum/$totalRuns] $runName ===" -ForegroundColor Cyan
    $cmd = @("-m", "diar_pipeline.run", "-i", $f.audio, "--reference-rttm", $f.ref,
             "--experiment", "ablate_enhance", "--run-name", $runName,
             "--embed", $BestEmbed, "--estimate", $BestEstimate,
             "--cluster", "sc", "--no-enhance") + $BestVad
    if ($f.transcribe) { $cmd += "--transcribe" }
    & $py @cmd
}

Write-Host ""
Write-Host "=== Phase 6 complete ===" -ForegroundColor Green
Write-Host "If enhance_off wins, drop --enhance in Phase 7+ by adding --no-enhance."
