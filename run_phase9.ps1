# Phase 9 - Silhouette refinement (1 setting x 4 files = 4 runs)
# Run from project root:  ./run_phase9.ps1

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

# <<< EDIT after Phases 3-8 >>>
$BestEmbed    = "resnet34"
$BestEstimate = "gmm_bic"
$BestCluster  = "sc"
$EnhanceFlag  = @()
$BestWindow   = @("--win-len", "1.5", "--hop-len", "0.75")
$BestVbx      = @()  # or: @("--refine-vbx") or @("--refine-vbx", "--vbx-Fa", "0.3", "--vbx-Fb", "25")
$BestVad      = @("--vad-model", "silero", "--vad-threshold", "0.4")

$totalRuns = $files.Count
$runNum = 0

foreach ($f in $files) {
    $runNum++
    $runName = "$($f.name)_silhouette"
    Write-Host ""
    Write-Host "=== [$runNum/$totalRuns] $runName ===" -ForegroundColor Cyan

    $cmd = @("-m", "diar_pipeline.run", "-i", $f.audio, "--reference-rttm", $f.ref,
             "--experiment", "ablate_silhouette", "--run-name", $runName,
             "--embed", $BestEmbed, "--estimate", $BestEstimate,
             "--cluster", $BestCluster, "--silhouette-refine") + $BestWindow + $BestVad + $EnhanceFlag + $BestVbx
    if ($f.transcribe) { $cmd += "--transcribe" }

    & $py @cmd
}

Write-Host ""
Write-Host "=== Phase 9 complete ===" -ForegroundColor Green
Write-Host "Compare against the Phase 8 winner to decide if silhouette helps."
Write-Host "Combine ALL winners into BEST_CONFIG and edit run_phase10.ps1 with them."
