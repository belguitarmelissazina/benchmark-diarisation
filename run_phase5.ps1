# Phase 5 - Clustering algorithm ablation (3 clusterers x 4 files = 12 runs)
# Run from project root:  ./run_phase5.ps1

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

# <<< EDIT after Phases 3 & 4 >>>
$BestEmbed    = "resnet34"
$BestEstimate = "gmm_bic"
$BestVad      = @("--vad-model", "silero", "--vad-threshold", "0.4")

$clusters = @("sc", "ahc", "meanshift")

$totalRuns = $clusters.Count * $files.Count
$runNum = 0

foreach ($cluster in $clusters) {
    foreach ($f in $files) {
        $runNum++
        $runName = "$($f.name)_$cluster"
        Write-Host ""
        Write-Host "=== [$runNum/$totalRuns] $runName ===" -ForegroundColor Cyan

        $cmd = @("-m", "diar_pipeline.run", "-i", $f.audio, "--reference-rttm", $f.ref,
                 "--experiment", "ablate_cluster", "--run-name", $runName,
                 "--embed", $BestEmbed, "--estimate", $BestEstimate,
                 "--cluster", $cluster) + $BestVad
        if ($f.transcribe) { $cmd += "--transcribe" }

        & $py @cmd
    }
}

Write-Host ""
Write-Host "=== Phase 5 complete ===" -ForegroundColor Green
Write-Host "Pick the lowest-DER clustering as BEST_CLUSTER for Phase 6+."
Write-Host "Note: Phase 6 (enhance) only runs if BEST_CLUSTER == sc."
