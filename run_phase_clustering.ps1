# Phase: Clustering method comparison — self-calibrating (percentile-based)
# Only percentile sweeps; absolute thresholds dropped (not portable across audio).
# Baseline SC+gmm_bic is NOT rerun — compare against existing MLflow runs.
# Run from project root:  ./run_phase_clustering.ps1

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

$BestEmbed = "resnet34"
$BestVad   = @("--vad-model", "silero", "--vad-threshold", "0.4")

# AHC percentile of cosine-distance distribution. Lower = stricter = more speakers.
# Empirically (see diagnostic on AMI-EN): p<40 over-segments, p>=50 merges into 1.
$ahcPercentiles    = @(40, 45, 50)
# Greedy percentile of cosine-similarity distribution. Higher = stricter = more speakers.
$greedyPercentiles = @(50, 55, 60)

$totalRuns = ($ahcPercentiles.Count + $greedyPercentiles.Count) * $files.Count
$runNum = 0

Write-Host "============================================================" -ForegroundColor Yellow
Write-Host "  Clustering comparison (percentile-based): $totalRuns runs" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow

# AHC percentile sweep (self-calibrating)
foreach ($p in $ahcPercentiles) {
    foreach ($f in $files) {
        $runNum++
        $runName = "$($f.name)_ahc_p$p"
        Write-Host ""
        Write-Host "=== [$runNum/$totalRuns] $runName ===" -ForegroundColor Cyan

        $cmd = @(
            "-m", "diar_pipeline.run",
            "-i", $f.audio,
            "--reference-rttm", $f.ref,
            "--experiment", "ablate_clustering_v2",
            "--run-name", $runName,
            "--embed", $BestEmbed,
            "--cluster", "ahc_threshold",
            "--ahc-percentile", "$p"
        ) + $BestVad
        if ($f.transcribe) { $cmd += "--transcribe" }
        & $py @cmd
    }
}

# Cosine greedy percentile sweep (self-calibrating)
foreach ($p in $greedyPercentiles) {
    foreach ($f in $files) {
        $runNum++
        $runName = "$($f.name)_greedy_p$p"
        Write-Host ""
        Write-Host "=== [$runNum/$totalRuns] $runName ===" -ForegroundColor Cyan

        $cmd = @(
            "-m", "diar_pipeline.run",
            "-i", $f.audio,
            "--reference-rttm", $f.ref,
            "--experiment", "ablate_clustering_v2",
            "--run-name", $runName,
            "--embed", $BestEmbed,
            "--cluster", "cosine_greedy",
            "--greedy-percentile", "$p"
        ) + $BestVad
        if ($f.transcribe) { $cmd += "--transcribe" }
        & $py @cmd
    }
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Clustering comparison complete" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Compare ablate_clustering runs against existing SC+gmm_bic baselines."
